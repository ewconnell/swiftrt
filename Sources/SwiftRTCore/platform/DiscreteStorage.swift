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
/// DiscreteStorage
public final class DiscreteStorage: StorageBuffer {
    // StorageBuffer protocol properties
    public let alignment: Int
    public let byteCount: Int
    public let id: Int
    public var isReadOnly: Bool
    public var isReference: Bool
    public var name: String

    //--------------------------------------------------------------------------
    // implementation properties
    
    /// the last queue used to mutate the storage
    public var lastMutatingQueueId: Int
    /// the index of the last memory buffer written to
    public var master: Int
    /// replicated device memory buffers
    public var replicas: [DeviceMemory?]

    //--------------------------------------------------------------------------
    // init(type:count:layout:name:
    @inlinable public init<Element>(
        storedType: Element.Type,
        count: Int,
        name: String = "Tensor"
    ) {
        self.name = name
        alignment = MemoryLayout<Element>.alignment
        byteCount = MemoryLayout<Element>.size * count
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false

        // setup replica managment
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
    // init(copying other:
    @inlinable public init(
        copying other: DiscreteStorage,
        using queue: DeviceQueue
    ) {
        id = Context.nextBufferId
        alignment = other.alignment
        byteCount = other.byteCount
        isReadOnly = other.isReadOnly
        isReference = other.isReference
        name = other.name

        // setup replica managment
        master = -1
        lastMutatingQueueId = 0
        let numDevices = Context.local.platform.devices.count
        replicas = [DeviceMemory?](repeating: nil, count: numDevices)

        // copy other master to self using the current queue
        
        
    }
    
    //--------------------------------------------------------------------------
    // init(buffer:layout:
    @inlinable public convenience init<Element>(
        referenceTo buffer: UnsafeBufferPointer<Element>
    ) {
        self.init(storedType: Element.self, count: buffer.count,
                  name: "Reference Tensor")
        isReadOnly = true
        isReference = true
        let p = UnsafeMutableBufferPointer(mutating: buffer)
        let raw = UnsafeMutableRawBufferPointer(p)
        replicas[0] = DeviceMemory(deviceId: 0, buffer: raw, type: .unified)
    }
    
    //--------------------------------------------------------------------------
    // init(type:buffer:layout:
    @inlinable public convenience init<Element>(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>
    ) {
        self.init(storedType: Element.self, count: buffer.count,
                  name: "Reference Tensor")
        isReference = true
        let raw = UnsafeMutableRawBufferPointer(buffer)
        replicas[0] = DeviceMemory(deviceId: 0, buffer: raw, type: .unified)
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
        using queue: DeviceQueue
    ) -> UnsafeBufferPointer<Element>
    {
        let start = getMemory(queue).buffer.baseAddress!
                .bindMemory(to: Element.self, capacity: count)
                .advanced(by: index)
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: DeviceQueue
    ) -> UnsafeMutableBufferPointer<Element> {
        let start = getMemory(queue).buffer.baseAddress!
                .bindMemory(to: Element.self, capacity: count)
                .advanced(by: index)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    @inlinable public func getMemory(_ queue: DeviceQueue) -> DeviceMemory {
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
                let memory = try queue.allocate(byteCount: byteCount)
                replicas[queue.deviceId] = memory
                
                // the new buffer is now the master version
                master = queue.deviceId
                return memory
            } catch {
                // Fail for now
                writeLog("Failed to allocate memory on \(queue.deviceName)")
                fatalError("TODO: implement LRU host migration" +
                            " and discrete memory discard")
            }
        }
    }
}
