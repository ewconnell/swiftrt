//******************************************************************************
// Copyright 2019 Google LLC
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
/// TensorArray
/// The TensorArray object is a flat array of values used by the TensorView.
/// It is responsible for replication and syncing between devices.
/// It is not created or directly used by end users.
public final class TensorArray<Element>: ObjectTracking, Codable, Logging
    where Element: TensorElementConformance
{
    //--------------------------------------------------------------------------
    /// the number of elements in the data array
    public let count: Int
    /// `true` if the data array references an existing read only buffer
    public let isReadOnly: Bool
    /// testing: `true` if the last access caused the contents of the
    /// buffer to be copied
    public var lastAccessCopiedBuffer: Bool
    /// the last queue id that wrote to the tensor
    public var lastMutatingQueue: DeviceQueue?
    /// whenever a buffer write pointer is taken, the associated DeviceArray
    /// becomes the master copy for replication. Synchronization across threads
    /// is still required for taking multiple write pointers, however
    /// this does automatically synchronize data migrations.
    /// The value will be `nil` if no access has been taken yet
    @usableFromInline
    var master: DeviceArray?
    /// this is incremented each time a write pointer is taken
    /// all replicated buffers will stay in sync with this version
    @usableFromInline
    var masterVersion: Int
    /// name label used for logging
    public let name: String
    /// replication collection
    @usableFromInline
    var replicas: [Int : DeviceArray]
    /// the object tracking id
    public let trackingId: Int

    //--------------------------------------------------------------------------
    // common
    @inlinable
    public init(count: Int, isReadOnly: Bool, name: String) {
        self.count = count
        self.isReadOnly = isReadOnly
        self.lastAccessCopiedBuffer = false
        self.masterVersion = -1
        self.name = name
        self.replicas = [Int : DeviceArray]()
        self.trackingId = ObjectTracker.global.nextId
        #if DEBUG
        ObjectTracker.global.register(
            self, namePath: logNamePath, supplementalInfo:
                "\(String(describing: Element.self))[\(count)]")

        diagnostic("\(createString) \(name)(\(trackingId)) " +
            "\(String(describing: Element.self))[\(count)]",
            categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // empty
    @inlinable
    public convenience init() {
        self.init(count: 0, isReadOnly: false, name: "")
    }

    //--------------------------------------------------------------------------
    // casting used for safe conversion between FixedSizeVector and Scalar
    @inlinable
    public convenience init<T>(_ other: TensorArray<T>) where
        T: FixedSizeVector, T.Scalar == Element
    {
        self.init(count: other.count * T.count, isReadOnly: false, name: other.name)
        self.replicas = other.replicas
    }

    //--------------------------------------------------------------------------
    // create a new element array
    @inlinable
    public convenience init(count: Int, name: String) {
        self.init(count: count, isReadOnly: false, name: name)
    }

    //--------------------------------------------------------------------------
    // create a new element array initialized with values
    @inlinable
    public convenience init<C>(elements: C, name: String) where
        C: Collection, C.Element == Element
    {
        self.init(count: elements.count, isReadOnly: false, name: name)
        
        // this should never fail since it is copying from host buffer to
        // host buffer. It is synchronous, so we don't need to create or
        // record a completion event.
        let buffer = readWrite(using: globalPlatform.transferQueue)
        for i in zip(buffer.indices, elements.indices) {
            buffer[i.0] = elements[i.1]
        }
    }
    
    //--------------------------------------------------------------------------
    // All initializers copy the data except this one which creates a
    // read only reference to avoid unnecessary copying from the source
    @inlinable
    public convenience init(referenceTo buffer: UnsafeBufferPointer<Element>,
                            name: String)
    {
        self.init(count: buffer.count, isReadOnly: true, name: name)
        masterVersion = 0
        
        // create the replica device array
        let platform = globalPlatform
        let queue = platform.currentQueue
        let key = queue.arrayReplicaKey
        let bytes = UnsafeRawBufferPointer(buffer)
        let array = queue.device.createReferenceArray(buffer: bytes)
        array.version = -1
        replicas[key] = array
    }
    
    //--------------------------------------------------------------------------
    /// uses the specified UnsafeMutableBufferPointer as the host
    /// backing stored
    @inlinable
    public convenience init(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>,
        name: String)
    {
        self.init(count: buffer.count, isReadOnly: false, name: name)
        masterVersion = 0
        
        // create the replica device array
        let queue = globalPlatform.currentQueue
        let key = queue.arrayReplicaKey
        let bytes = UnsafeMutableRawBufferPointer(buffer)
        let array = queue.device.createMutableReferenceArray(buffer: bytes)
        array.version = -1
        replicas[key] = array
    }
    
    //--------------------------------------------------------------------------
    // init from other TensorArray
    @inlinable
    public convenience init(copying other: TensorArray,
                            using queue: DeviceQueue)
    {
        self.init(count: other.count, isReadOnly: other.isReadOnly,
                  name: other.name)
        masterVersion = 0

        // make sure there is something to copy
        guard let otherMaster = other.master else { return }
        
        // get the array replica for `queue`
        let replica = getArray(for: queue)
        replica.version = masterVersion
        
        // copy the other master array
        queue.copyAsync(to: replica, from: otherMaster)

        diagnostic("\(copyString) \(name)(\(trackingId)) " +
            "\(otherMaster.deviceName)" +
            "\(setText(" --> ", color: .blue))" +
            "\(queue.deviceName)_q\(queue.id) " +
            "\(String(describing: Element.self))[\(count)]",
            categories: .dataCopy)
    }

    //--------------------------------------------------------------------------
    @inlinable
    deinit {
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        if count > 0 {
            diagnostic("\(releaseString) \(name)(\(trackingId)) ",
                categories: .dataAlloc)
        }
        #endif
    }

    //--------------------------------------------------------------------------
    /// readOnly
    /// - Parameter queue: the queue to use for synchronizatoin and locality
    /// - Returns: an Element buffer
    public func readOnly(using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    {
        UnsafeBufferPointer(migrate(readOnly: true, using: queue))
    }
    
    //--------------------------------------------------------------------------
    /// readWrite
    /// - Parameter queue: the queue to use for synchronizatoin and locality
    /// - Returns: an Element buffer
    public func readWrite(using queue: DeviceQueue) ->
        UnsafeMutableBufferPointer<Element>
    {
        assert(!isReadOnly, "the TensorArray is read only")
        lastMutatingQueue = queue
        return migrate(readOnly: false, using: queue)
    }
    
    //--------------------------------------------------------------------------
    /// migrate
    /// This migrates the master version of the data from wherever it is to
    /// the device associated with `queue` and returns a pointer to the data
    private func migrate(readOnly: Bool, using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<Element>
    {
        // get the array replica for `queue`
        // this is a synchronous operation independent of queues
        let replica = getArray(for: queue)
        lastAccessCopiedBuffer = false

        // compare with master and copy if needed
        if let master = master, replica.version != master.version {
//            // cross service?
//            if replica.device.service.id != master.device.service.id {
//                copyCrossService(to: replica, from: master, using: queue)
//
//            } else
            if replica.deviceId != master.deviceId {
                copyCrossDevice(to: replica, from: master, using: queue)
            }
        }
        
        // set version
        if !readOnly { master = replica; masterVersion += 1 }
        replica.version = masterVersion
        return replica.buffer.bindMemory(to: Element.self)
    }

    //--------------------------------------------------------------------------
    // copyCrossService
    // copies from an array in one service to another
    private func copyCrossService(to other: DeviceArray,
                                  from master: DeviceArray,
                                  using queue: DeviceQueue)
    {
        lastAccessCopiedBuffer = true
        
        if master.memoryAddressing == .unified {
            // copy host to discreet memory device
            if other.memoryAddressing == .discreet {
                // get the master uma buffer
                let buffer = UnsafeRawBufferPointer(master.buffer)
                queue.copyAsync(to: other, from: buffer)

                diagnostic("\(copyString) \(name)(\(trackingId)) " +
                    "\(master.deviceName)\(setText(" --> ", color: .blue))" +
                    "\(other.deviceName)_q\(queue.id) " +
                    "\(String(describing: Element.self))[\(count)]",
                    categories: .dataCopy)
            }
            // otherwise they are both unified, so do nothing
        } else if other.memoryAddressing == .unified {
            // device to host
            queue.copyAsync(to: other.buffer, from: master)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(master.deviceName)_q\(queue.id)" +
                "\(setText(" --> ", color: .blue))\(other.deviceName) " +
                "\(String(describing: Element.self))[\(count)]",
                categories: .dataCopy)

        } else {
            // both are discreet and not in the same service, so
            // transfer to host memory as an intermediate step
            let host = getArray(for: globalPlatform.transferQueue)
            queue.copyAsync(to: host.buffer, from: master)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(master.deviceName)_q\(queue.id)" +
                "\(setText(" --> ", color: .blue))\(other.deviceName)" +
                "\(String(describing: Element.self))[\(count)]",
                categories: .dataCopy)
            
            let hostBuffer = UnsafeRawBufferPointer(host.buffer)
            queue.copyAsync(to: other, from: hostBuffer)
            
            diagnostic("\(copyString) \(name)(\(trackingId)) " +
                "\(other.deviceName)" +
                "\(setText(" --> ", color: .blue))" +
                "\(master.deviceName)_q\(queue.id) " +
                "\(String(describing: Element.self))[\(count)]",
                categories: .dataCopy)
        }
    }
    
    //--------------------------------------------------------------------------
    // copyCrossDevice
    // copies from one discreet memory device to the other
    private func copyCrossDevice(to other: DeviceArray,
                                 from master: DeviceArray,
                                 using queue: DeviceQueue)
    {
        // only copy if the devices do not have unified memory
        guard master.memoryAddressing == .discreet else { return }
        lastAccessCopiedBuffer = true
        
        // async copy and record completion event
        queue.copyAsync(to: other, from: master)

        diagnostic("\(copyString) \(name)(\(trackingId)) " +
            "\(master.deviceName)" +
            "\(setText(" --> ", color: .blue))" +
            "\(queue.deviceName)_q\(queue.id) " +
            "\(String(describing: Element.self))[\(count)]",
            categories: .dataCopy)
    }
    
    //--------------------------------------------------------------------------
    // getArray(queue:
    // This manages a dictionary of replicated device arrays indexed
    // by serviceId and id. It will lazily create a device array if needed
    @inlinable
    public func getArray(for queue: DeviceQueue) -> DeviceArray {
        // lookup array associated with this queue
        let key = queue.arrayReplicaKey
        if let replica = replicas[key] {
            return replica
        } else {
            // create the replica device array
            let byteCount = MemoryLayout<Element>.size * count
            let array = queue.device.createArray(byteCount: byteCount,
                                                     heapIndex: 0,
                                                     zero: false)
            diagnostic("\(allocString) \(name)(\(trackingId)) " +
                "device array on \(queue.deviceName) " +
                "\(String(describing: Element.self))[\(count)]",
                categories: .dataAlloc)
            
            array.version = -1
            replicas[key] = array
            return array
        }
    }

    //==========================================================================
    // Codable
    // useful discussion on techniques
    // https://www.raywenderlich.com/3418439-encoding-and-decoding-in-swift
    public enum CodingKeys: String, CodingKey { case name, data }
    
    /// encodes the contents of the array
    @inlinable
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        var dataContainer = container.nestedUnkeyedContainer(forKey: .data)
        let buffer = readOnly(using: globalPlatform.transferQueue)
        try buffer.forEach {
            try dataContainer.encode($0)
        }
    }
    
    @inlinable
    public convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        var dataContainer = try container.nestedUnkeyedContainer(forKey: .data)
        if let count = dataContainer.count {
            self.init(count: count, name: name)
            let elements = readWrite(using: globalPlatform.transferQueue)
            for i in 0..<count {
                elements[i] = try dataContainer.decode(Element.self)
            }
        } else {
            self.init(count: 0, name: name)
        }
    }
}
