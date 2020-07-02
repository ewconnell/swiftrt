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
import CCuda
import Numerics
//import CudaKernels

//==============================================================================
/// CudaQueue
public final class CudaQueue: DeviceQueue, CpuFunctions {
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceIndex: Int
    public let id: Int
    public let logInfo: LogInfo
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
        logInfo: LogInfo,
        name: String,
        queueMode: DeviceQueueMode,
        useGpu: Bool
    ) {
        do {
            self.id = Context.nextQueueId
            self.name = name
            self.logInfo = logInfo.flat(name)
            self.deviceIndex = deviceIndex
            self.gpuId = deviceIndex - 1
            self.creatorThread = Thread.current
            self.defaultQueueEventOptions = QueueEventOptions()
            self.memoryType = useGpu ? .discrete : .unified
            self.mode = queueMode
            self.queue = DispatchQueue(label: name)
            self.group = DispatchGroup()
            self.useGpu = useGpu
            
            // select the specified device
            try cudaCheck(status: cudaSetDevice(Int32(gpuId)))
            
            // create a queue associated with the device
            let flags = UInt32(cudaStreamNonBlocking)
            var cudaStream: cudaStream_t?
            try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
            stream = cudaStream!
            cudnn = CudnnHandle(gpuId: gpuId, using: stream)
            cublas = CublasHandle(gpuId: gpuId, using: stream)
        } catch {
            Context.currentDevice.writeLog("\(error)")
            fatalError()
        }

        diagnostic("\(createString) queue: \(name)", categories: .queueAlloc)
    }
    
    //--------------------------------------------------------------------------
    // select
    public func selectDevice() throws {
        try cudaCheck(status: cudaSetDevice(Int32(gpuId)))
    }
    
    //==========================================================================
    /// createActivation
//    public override func createActivation<T>(
//        x: T,
//        y: inout T,
//        mode: ActivationType,
//        nan: NanPropagation,
//        reluCeiling: Double = 0) throws -> ActivationInferring<T>
//        where T: TensorView, T.Element: ScalarElement & FloatingPoint
//    {
//        return try CudaActivationInferring(x: x, y: &y, mode: mode,
//                                           nan: nan, reluCeiling: reluCeiling)
//    }
}
