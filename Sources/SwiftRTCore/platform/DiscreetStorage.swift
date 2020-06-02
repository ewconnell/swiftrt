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

//==============================================================================
/// DiscreetStorage
public final class DiscreetStorage: StorageBuffer {
    // StorageBuffer protocol properties
    public let alignment: Int
    public let byteCount: Int
    public let id: Int
    public let isReadOnly: Bool
    public let isReference: Bool
    public let layout: Layout
    public var name: String

    //--------------------------------------------------------------------------
    // implementation properties
    
    /// the last queue used to mutate the storage
    public var lastMutatingQueueId: Int
    /// the index of the last memory buffer written to
    public var master: Int
    /// replicated device memory buffers
    public var replicas: [DeviceMemory?]

    /// the host transfer buffer
    @inlinable public var hostBuffer: UnsafeMutableRawBufferPointer {
        assert(master == 0, "`read` or `readWrite` on device 0" +
               " must be called prior to access")
        return replicas[0]!.buffer
    }

    //--------------------------------------------------------------------------
    // init(type:count:layout:name:
    @inlinable public init<Element>(
        storedType: Element.Type,
        count: Int,
        layout: Layout,
        name: String = "Tensor"
    ) {
        self.layout = layout
        self.name = name
        alignment = MemoryLayout<Element>.alignment
        byteCount = MemoryLayout<Element>.size * count
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false

        master = -1
        lastMutatingQueueId = 0
        let numDevices = Context.local.platform.devices.count
        replicas = [DeviceMemory?](repeating: nil, count: numDevices)

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
                    "\(Element.self)[\(count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(other:
    @inlinable public init(copying other: DiscreetStorage) {
        alignment = other.alignment
        layout = other.layout
        byteCount = other.byteCount
        id = Context.nextBufferId
        isReadOnly = other.isReadOnly
        isReference = other.isReference
        name = other.name
        fatalError()

//        if isReference {
//            hostBuffer = other.hostBuffer
//        } else {
//            hostBuffer = UnsafeMutableRawBufferPointer.allocate(
//                byteCount: other.byteCount,
//                alignment: other.alignment)
//            hostBuffer.copyMemory(from: UnsafeRawBufferPointer(other.hostBuffer))
//        }
    }
    
    //--------------------------------------------------------------------------
    // init(buffer:layout:
    @inlinable public init<Element>(
        referenceTo buffer: UnsafeBufferPointer<Element>,
        layout: Layout
    ) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // init(type:buffer:layout:
    @inlinable public init<Element>(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>,
        layout: Layout
    ) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public init<S, Stream>(
        block shape: S,
        bufferedBlocks: Int,
        stream: Stream
    ) where S : TensorShape, Stream : BufferStream {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func read<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeBufferPointer<Element>
    {
        fatalError()
//        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
//        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite<Element>(
        type: Element.Type,
        at base: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
//        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
//        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    @inlinable public func getMemory(
        _ queue: PlatformType.Device.Queue
    ) -> DeviceMemory {
        if let memory = replicas[queue.deviceId] {
            if memory.version == replicas[master]!.version {
                return memory
            } else {
                // migrate
                fatalError()
            }
        } else {
            do {
                // allocate the buffer for the target device
                // and save in the replica list
                let memory = try queue.allocate(alignment: alignment,
                                                byteCount: byteCount,
                                                heapIndex: 0)
                replicas[queue.deviceId] = memory
                
                // the new buffer is now the master version
                master = queue.deviceId
                return memory
            } catch {
                // Fail for now
                writeLog("Failed to allocate memory on \(queue.deviceName)")
                fatalError("TODO: implement LRU host migration" +
                            " and discreet memory discard")
            }
        }
    }
}
