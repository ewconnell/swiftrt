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

    //------------------------------------
    // private properties
    /// replicated device memory buffers
    public var replicas: [DeviceMemory?]

    /// the last queue used to mutate the storage
    public var lastMutatingQueue: DeviceQueue?

    /// whenever a buffer write pointer is taken, the associated DeviceArray
    /// becomes the master copy for replication. Synchronization across threads
    /// is still required for taking multiple write pointers, however
    /// this does automatically synchronize data migrations.
    /// The value will be `nil` if no access has been taken yet
    public var master: DeviceMemory?

    /// this is incremented each time a write pointer is taken
    /// all replicated buffers will stay in sync with this version
    public var masterVersion: Int

    //------------------------------------
    // testing properties
    
    /// testing: `true` if the last access caused the contents of the
    /// buffer to be copied
    @usableFromInline var lastAccessCopiedBuffer = false
    /// testing: is `true` if the last data access caused the view's underlying
    /// tensorArray object to be copied. It's stored here instead of on the
    /// view, because the view is immutable when taking a read only pointer
    @usableFromInline var lastAccessMutatedView: Bool = false

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
        masterVersion = -1

        // setup replica managment
        let numDevices = Context.local.platform.devices.count
        replicas = [DeviceMemory?](repeating: nil, count: numDevices)
    }
    
    //--------------------------------------------------------------------------
    /// `init(storedElement:name:
    public convenience init<Element>(storedElement: Element, name: String) {
        // TODO: change this to cache single scalars
        self.init(storedType: Element.self, count: 1, name: name)
        let buffer = readWrite(type: Element.self, at: 0, count: 1,
                               using: Context.syncQueue)
        buffer[0] = storedElement
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
        masterVersion = -1
        
        // setup replica managment
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
        let device = Context.devices[0]
        replicas[0] = CpuDeviceMemory(device.id, device.name,
                                      buffer: raw, isReference: true)
        diagnostic(
            "\(referenceString) \(name)(\(id)) " +
                "\(Element.self)[\(buffer.count)]",
            categories: .dataAlloc)
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
        let device = Context.devices[0]
        replicas[0] = CpuDeviceMemory(device.id, device.name,
                                      buffer: raw, isReference: true)
        diagnostic(
            "\(referenceString) \(name)(\(id)) " +
                "\(Element.self)[\(buffer.count)]",
            categories: .dataAlloc)
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
    ) -> UnsafeBufferPointer<Element> {
        let buffer = migrate(type, readOnly: true, using: queue)
        assert(index + count <= buffer.count)
        let start = buffer.baseAddress!.advanced(by: index)
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
        let buffer = migrate(type, readOnly: false, using: queue)
        lastMutatingQueue = queue
        assert(index + count <= buffer.count)
        let start = buffer.baseAddress!.advanced(by: index)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }

    //--------------------------------------------------------------------------
    /// waitForCompletion
    /// blocks the caller until pending write operations have completed
    @inlinable public func waitForCompletion() {
        lastMutatingQueue?.waitUntilQueueIsComplete()
    }

    //--------------------------------------------------------------------------
    /// synchronize
    /// If the queue is changing, then this creates an event and
    /// records it onto the end of the lastQueue, then records a wait
    /// on the new queue. This insures storage mutations from the lastQueue
    /// finishes before the new one begins.
    @inlinable public func synchronize(with queue: DeviceQueue) throws {
        if let lastQueue = lastMutatingQueue,
           lastQueue.mode == .async && queue.id != lastQueue.id
        {
            let event = lastQueue.createEvent()
            diagnostic(
                "\(queue.deviceName)_\(queue.name) will wait for " +
                    "\(lastQueue.deviceName)_\(lastQueue.name) " +
                    "using event(\(event.id))",
                categories: .queueSync)
            queue.wait(for: lastQueue.record(event: event))
        }
    }
    
    //--------------------------------------------------------------------------
    /// migrate
    /// This migrates the master version of the data from wherever it is to
    /// the device associated with `queue`
    ///
    /// - Returns: a buffer pointer to the data
    @inlinable public func migrate<Element>(
        _ type: Element.Type,
        readOnly: Bool,
        using queue: DeviceQueue
    ) -> UnsafeMutableBufferPointer<Element> {
        do {
            // synchronize queue with last mutating queue
            try synchronize(with: queue)
            
            // Get the buffer for this tensor on the device associated
            // with `queue`. This is a synchronous operation.
            let replica = try getDeviceMemory(type, queue)
            
            // copy from the master to the replica if the versions don't match
            if let master = master,
               replica.version != master.version &&
                replica.deviceId != master.deviceId
            {
                // we shouldn't get here if both buffers are in unified memory
                assert(master.type == .discrete || replica.type == .discrete)
                lastAccessCopiedBuffer = true
                try queue.copyAsync(from: master, to: replica)
                diagnostic(
                    "\(copyString) \(name)(\(id)) " +
                        "\(master.deviceName)" +
                        "\(setText(" --> ", color: .blue))" +
                        "\(queue.deviceName)_s\(queue.id) " +
                        "\(Element.self)[\(replica.buffer.count)]",
                    categories: .dataCopy)
            }
            
            // set version
            if !readOnly { master = replica; masterVersion += 1 }
            replica.version = masterVersion
            return replica.buffer.bindMemory(to: Element.self)
        } catch {
            // Fail for now
            writeLog("Failed to access memory")
            fatalError()
        }
    }

    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    @inlinable public func getDeviceMemory<Element>(
        _ type: Element.Type,
        _ queue: DeviceQueue
    ) throws -> DeviceMemory {
        if let memory = replicas[queue.deviceId] {
            return memory
        } else {
                // allocate the buffer for the target device
                // and save in the replica list
                let memory = try queue.allocate(byteCount: byteCount)
                replicas[queue.deviceId] = memory
                
                if willLog(level: .diagnostic) {
                    let count = byteCount / MemoryLayout<Element>.size
                    diagnostic(
                        "\(allocString) \(name)(\(id)) " +
                            "\(Element.self)[\(count)] on \(queue.deviceName)",
                        categories: .dataAlloc)
                }

                // the new buffer is now the master version
                master = memory
                return memory
        }
    }
}
