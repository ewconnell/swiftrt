//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import SwiftRTCuda
import Numerics

//==============================================================================
/// CudaQueue
public final class CudaQueue: DeviceQueue, CpuFunctions {
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceIndex: Int
    public let id: Int
    public let memoryType: MemoryType
    public let mode: DeviceQueueMode
    public let name: String
    public let queue: DispatchQueue
    public let group: DispatchGroup
    public let useGpu: Bool
    
    public let gpuId: Int
    public let stream: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle
    @inlinable public var usesCpu: Bool { !useGpu }

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(
        deviceIndex: Int,
        name: String,
        queueMode: DeviceQueueMode,
        useGpu: Bool
    ) {
        self.id = Context.nextQueueId
        self.name = name
        self.deviceIndex = deviceIndex
        self.gpuId = Swift.max(0, deviceIndex - 1)
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.memoryType = useGpu ? .discrete : .unified
        self.mode = queueMode
        self.queue = DispatchQueue(label: name)
        self.group = DispatchGroup()
        self.useGpu = useGpu
        
        // select the specified device
        cudaCheck(cudaSetDevice(Int32(gpuId)))
        
        // create a queue associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        cudaCheck(cudaStreamCreateWithFlags(&cudaStream, flags))
        stream = cudaStream!
        cudnn = CudnnHandle(gpuId: gpuId, using: stream)
        cublas = CublasHandle()

        diagnostic(.create, "queue: \(name)", categories: .queueAlloc)
    }
    
    //--------------------------------------------------------------------------
    // select
    @inlinable public func selectDevice() {
        cudaCheck(cudaSetDevice(Int32(gpuId)))
    }
    
    //--------------------------------------------------------------------------
    // allocate
    // allocate a device memory buffer suitably aligned for any type
    @inlinable public func allocate(
        byteCount: Int,
        heapIndex: Int = 0
    ) -> DeviceMemory {
        if usesCpu {
            let buffer = UnsafeMutableRawBufferPointer
                    .allocate(byteCount: byteCount,
                              alignment: MemoryLayout<Int>.alignment)
            return CpuDeviceMemory(deviceIndex, buffer, memoryType)
        } else {
            return CudaDeviceMemory(deviceIndex, byteCount)
        }
    }

    //--------------------------------------------------------------------------
    // delay
    @inlinable public func delay(_ interval: TimeInterval) {
        guard useGpu else { cpu_delay(interval); return }
        srtDelayStream(interval, stream)
    }

    //--------------------------------------------------------------------------
    // copyAsync
    @inlinable public func copyAsync(
        from src: DeviceMemory, 
        to dst: DeviceMemory
    ) {
        assert(src.buffer.count == dst.buffer.count)

        switch (src.type, dst.type) {
        // host --> host
        case (.unified, .unified):
            cpu_copyAsync(from: src, to: dst)

        // host --> discrete
        case (.unified, .discrete):
            cudaCheck(cudaMemcpyAsync(
                dst.mutablePointer,	
                src.pointer,
                src.buffer.count,
                cudaMemcpyHostToDevice, 
                stream))

        // discrete --> host
        case (.discrete, .unified):
            cudaCheck(cudaMemcpyAsync(
                dst.mutablePointer,	
                src.pointer,
                src.buffer.count,
                cudaMemcpyDeviceToHost, 
                stream))

        // discrete --> discrete
        case (.discrete, .discrete):
            cudaCheck(cudaMemcpyAsync(
                dst.mutablePointer,	
                src.pointer,
                src.buffer.count,
                cudaMemcpyDeviceToDevice, 
                stream))
        }
    }

    //--------------------------------------------------------------------------
    @inlinable public func recordEvent() -> CudaEvent {
        let event = CudaEvent(recordedOn: self)
        if useGpu {
            cudaCheck(cudaEventRecord(event.handle, stream))
        } else {
            if mode == .async {
                queue.async(group: group) {
                    event.signal()
                }
            }
        }
        return event
    }
    
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// causes this queue to wait until the event has occurred
    @inlinable public func wait(for event: CudaEvent) {
        if useGpu {
            cudaCheck(cudaStreamWaitEvent(stream, event.handle, 0))
        } else {
            if mode == .sync {
                event.wait()
            } else {
                queue.async(group: group) {
                    event.wait()
                }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    // waitForCompletion
    // the synchronous queue completes work as it is queued,
    // so it is always complete
    @inlinable public func waitForCompletion() {
        if useGpu {
            cudaCheck(cudaStreamSynchronize(stream))
        } else if mode == .async {
            group.wait()
        }
    }
}
