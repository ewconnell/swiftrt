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

import Foundation

//==============================================================================
/// MultiDeviceBuffer
/// Used internally to manage the state of a collection of device buffers
public class MultiDeviceBuffer: ElementBuffer {
    public var id: Int
    public var isReference: Bool
    
    /// the number of bytes in the buffer
    public let byteCount: Int
    /// a dictionary of device memory replicas allocated on each device
    /// - Parameter key: the device index
    /// - Returns: the associated device memory object
    public var memory: [Int : DeviceMemory]

    /// `true` if the buffer is not mutable, such as in the case of a readOnly
    /// reference buffer.
    public let isReadOnly: Bool

    /// the `id` of the last queue that obtained write access
    public var lastMutatingQueue: QueueId

    /// the buffer name used in diagnostic messages
    public let name: String

    /// the index of the device holding the master version
    public var masterDevice: Int

    /// the masterVersion is incremented each time write access is taken.
    /// All device buffers will stay in sync with this version, copying as
    /// necessary.
    public var masterVersion: Int

    /// helper to return `Element` sized count
    @inlinable
    public func count<Element>(of type: Element.Type) -> Int {
        byteCount * MemoryLayout<Element>.size
    }

    //--------------------------------------------------------------------------
    // init(type:count:name:
    @inlinable
    public init<E>(type: E.Type, count: Int, name: String) {
        self.byteCount = count * MemoryLayout<E>.size
        self.memory = [Int : DeviceMemory]()
        self.id = Platform.nextBufferId
        self.isReference = false
        self.isReadOnly = false
        self.lastMutatingQueue = QueueId(0, 0)
        self.masterDevice = 0
        self.masterVersion = 0
        self.name = name
    }

    //--------------------------------------------------------------------------
    /// `deallocate`
    /// releases device memory associated with this buffer descriptor
    /// - Parameter device: the device to release memory from. `nil` will
    /// release all associated device memory for this buffer.
    @inlinable
    public func deallocate(device: Int? = nil) {
        if let device = device {
            memory[device]!.deallocate()
        } else {
            memory.values.forEach { $0.deallocate() }
        }
    }
    
    
    public func duplicate() -> ElementBuffer {
        fatalError()
    }
    
    public func read<E>(type: E.Type, at offset: Int, count: Int,
                        using queue: DeviceQueue) -> UnsafeBufferPointer<E>
    {
        fatalError()
    }
    
    public func readWrite<E>(type: E.Type, at offset: Int, count: Int,
                             willOverwrite: Bool, using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<E>
    {
        fatalError()
    }
}

//==============================================================================
/// DiscreetMemoryManagement
/// Compute services that manage asynchronous discreet devices
/// conform to this protocol
public protocol DiscreetMemoryManagement: MemoryManagement {
    /// a dictionary of device buffer entries indexed by the device
    /// number, and keyed by the id returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [Int : MultiDeviceBuffer] { get set }
}

public extension DiscreetMemoryManagement where Self: PlatformService {
    //--------------------------------------------------------------------------
    // createBuffer
    @inlinable
    func createBuffer<Element>(of type: Element.Type, count: Int, name: String)
        -> BufferRef
    {
        let buffer = MultiDeviceBuffer(type: type, count: count, name: name)
        deviceBuffers[buffer.id] = buffer
        return BufferRef(buffer)
    }

    //--------------------------------------------------------------------------
    // createBuffer
    @inlinable
    func createBuffer<E, Shape, Stream>(
        of type: E.Type, block shape: Shape,
        bufferedBlocks: Int, stream: Stream) -> (BufferRef, Int)
        where Shape : ShapeProtocol, Stream : BufferStream
    {
        fatalError()
    }

    //--------------------------------------------------------------------------
    // cachedBuffer
    @inlinable
    func cachedBuffer<Element>(for element: Element) -> BufferRef
    {
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createReference
    // create the DeviceBuffer record and add it to the dictionary
    @inlinable
    func createReference<Element>(to buffer: UnsafeBufferPointer<Element>,
                                  name: String) -> BufferRef
    {
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createMutableReference
    // create the DeviceBuffer record and add it to the dictionary
    @inlinable
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferRef
    {
        fatalError()
    }

    //--------------------------------------------------------------------------
    // duplicate
    @inlinable
    func duplicate(_ ref: BufferRef, using queue: QueueId) -> BufferRef {
        fatalError()
    }

    //--------------------------------------------------------------------------
    // release
    @inlinable
    func release(_ ref: BufferRef) {
        deviceBuffers[ref.id]!.deallocate()
    }

    //--------------------------------------------------------------------------
    // read
    @inlinable
    func read<Element>(_ ref: BufferRef, of type: Element.Type,
                       at offset: Int, count: Int, using queueId: QueueId)
        -> UnsafeBufferPointer<Element>
    {
        assert(deviceBuffers[ref.id] != nil)
        let buffer = migrate(ref, of: type, readOnly: true, using: queueId)
        return UnsafeBufferPointer(
            start: buffer.baseAddress!.advanced(by: offset),
            count: count)
    }

    //--------------------------------------------------------------------------
    // readWrite
    @inlinable
    func readWrite<Element>(_ ref: BufferRef, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queueId: QueueId)
        -> UnsafeMutableBufferPointer<Element>
    {
        assert(deviceBuffers[ref.id] != nil)
        // record the mutating queueId
        deviceBuffers[ref.id]!.lastMutatingQueue = queueId
        let buffer = migrate(ref, of: type, readOnly: false, using: queueId)
        return UnsafeMutableBufferPointer(
            start: buffer.baseAddress!.advanced(by: offset),
            count: count)
    }

    //--------------------------------------------------------------------------
    /// migrate
    /// Migrates the master version of the data from wherever it is to
    /// the device associated with `queue` and returns a pointer to the data
    @inlinable
    func migrate<Element>(_ ref: BufferRef, of type: Element.Type,
                          readOnly: Bool, using queueId: QueueId)
        -> UnsafeMutableBufferPointer<Element>
    {
        // get a reference to the device buffer
        let device = queueId.device
        var deviceMemory = getDeviceMemory(ref, of: type, for: device)

        //        // compare with master and copy if needed
        //        if let master = buffer.masterDevice,
        //            replica.version != buffer.masterVersion {
        //            // cross service?
        //            if replica.device.service.id != master.device.service.id {
        //                try copyCrossService(to: replica, from: master, using: queue)
        //
        //            } else if replica.device.id != master.device.id {
        //                try copyCrossDevice(to: replica, from: master, using: queue)
        //            }
        //        }

        // set version
        if !readOnly {
            deviceBuffers[ref.id]!.masterDevice = device
            deviceBuffers[ref.id]!.masterVersion += 1
        }
        deviceMemory.version = deviceBuffers[ref.id]!.masterVersion
        deviceBuffers[ref.id]!.memory[device] = deviceMemory

        return deviceMemory.buffer.bindMemory(to: Element.self)
    }

    //--------------------------------------------------------------------------
    // getDeviceMemory(_:ref:type:device:
    // returns the device memory buffer associated with the specified
    // queueId. It will lazily create the memory if needed.
    @inlinable
    func getDeviceMemory<Element>(_ ref: BufferRef, of type: Element.Type,
                                  for device: Int) -> DeviceMemory
    {
        // get a reference to the device buffer
        let buffer = deviceBuffers[ref.id]!

        // if the memory exists then return it
        if let deviceMemory = buffer.memory[device] {
            return deviceMemory
        } else {
            // allocate the memory on the target device
            let deviceMemory = devices[device]
                .allocate(byteCount: buffer.byteCount, heapIndex: 0)

            diagnostic("\(allocString) \(name)(\(ref.id)) " +
                "device array on \(devices[device].name) \(Element.self)" +
                "[\(buffer.count(of: Element.self))]",
                categories: .dataAlloc)
            return deviceMemory
        }
    }
}
