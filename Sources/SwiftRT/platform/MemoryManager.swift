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
/// MemoryManagement
public protocol MemoryManagement: class {
    /// a dictionary of device buffer entries indexed by the device
    /// number, and keyed by the id returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [Int : DeviceBuffer] { get set }
    
    /// generates a unique buffer reference
    var nextBufferRef: BufferRef { get }
    
    //--------------------------------------------------------------------------
    /// `bufferName(ref:`
    /// - Parameter ref: the buffer reference object
    /// - Returns: the name of the buffer used in diagnostic messages
    func bufferName(_ ref: BufferRef) -> String
    
    /// `createBuffer(type:count:`
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// A `BufferRef` is used so the associated memory can be moved by the
    /// service between accesses in order to maximize memory utilization.
    /// - Parameter type: the element type of the buffer
    /// - Parameter count: size of the buffer in `Element` units
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createBuffer<Element>(of type: Element.Type, count: Int,
                               name: String) -> BufferRef

    /// `createBuffer(blockSize:bufferedBlocks:sequence:`
    /// creates a streaming device buffer to be used in tensor operations.
    /// - Parameter shape: the shape of the blocks read or written to
    /// the sequence in a given transaction. This might be the number
    /// of elements in a view.
    /// - Parameter bufferedBlocks: the size of the device buffer
    /// to reserve in block units
    /// - Parameter stream: the I/O object for read/write operations
    /// - Returns: a buffer id and the size of the stream in block units.
    /// An endless sequence will return infinity for the block count.
    func createBuffer<Shape, Stream>(block shape: Shape,
                                     bufferedBlocks: Int,
                                     stream: Stream) -> (BufferRef, Int)
        where Shape: ShapeProtocol, Stream: BufferStream

    /// `cachedBuffer(element:`
    /// returns a device buffer initialized with the specified `Element`
    /// value. User expressions use a lot of constant scalar values
    /// which are repeated. For example: `let m = matrix + 1`. These
    /// expressions are frequently iterated thousands of times. This function
    /// will maintain a cache of constant values, which are likely to
    /// already be present on a discreet accelerator device,
    /// saving a lot of time.
    /// - Parameter element: the element value to cache
    /// - Returns: a device buffer reference that contains the element value.
    /// A BufferRef is created if it does not already exist.
    func cachedBuffer<Element>(for element: Element) -> BufferRef

    /// `createReference(buffer:`
    /// creates a device buffer whose data is associated with
    /// the specified buffer pointer. No memory is allocated, so the
    /// buffer must point to valid data space managed by the application.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter buffer: a buffer pointer to the data
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createReference<Element>(
        to buffer: UnsafeBufferPointer<Element>,
        name: String) -> BufferRef

    /// `createMutableReference(buffer:`
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferRef

    /// `duplicate(other:queue:`
    /// makes a duplicate of the specified device buffer. Used to support
    /// copy-on-write semantics
    /// - Parameter other: the id of the other device buffer to duplicate
    /// - Parameter queue: specifies the device/queue for synchronization.
    /// - Returns: a reference to the device buffer
    func duplicate(_ other: BufferRef, using queue: QueueId) -> BufferRef

    /// `release(buffer:`
    /// Releases a buffer created by calling `createBuffer`
    /// - Parameter buffer: the device buffer to release
    func release(_ ref: BufferRef)

    /// `read(ref:type:offset:queue:`
    /// - Parameter ref: reference to the device buffer to read
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queueId: queue id for data placement and synchronization
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized
    func read<Element>(_ ref: BufferRef, of type: Element.Type,
                       at offset: Int, count: Int,
                       using queueId: QueueId) -> UnsafeBufferPointer<Element>

    /// `readWrite(ref:type:offset:queue:willOverwrite:`
    /// - Parameter ref: eference to the device buffer to readWrite
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Parameter queueId: queue id for data placement and synchronization
    /// - Returns: a mutable buffer pointer to the elements associated with the
    /// specified buffer id. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite<Element>(_ ref: BufferRef, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queueId: QueueId)
        -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
/// BufferRef
/// a class object used to maintain a reference count to a set of
/// associated device buffers
public class BufferRef: Equatable {
    /// used to identify the `DeviceBuffer` instance
    public let id: Int

    /// initializer
    @inlinable public init(_ id: Int) { self.id = id }

    /// a buffer name used in diagnostic messages
    @inlinable
    public var name: String { Platform.service.bufferName(self) }

    // Equatable conformance
    public static func == (lhs: BufferRef, rhs: BufferRef) -> Bool {
        lhs.id == rhs.id
    }
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }

}

//==============================================================================
/// DeviceBuffer
/// Used internally to manage the state of a collection of device buffers
public struct DeviceBuffer {
    /// the number of bytes in the buffer
    public let byteCount: Int
    /// a dictionary of device memory replicas allocated on each device
    /// - Parameter key: the device index
    /// - Returns: the associated device memory object
    public var replicas: [Int : DeviceMemory]
    
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
    /// initializer
    @inlinable
    public init(byteCount: Int, name: String, isReadOnly: Bool = false) {
        self.byteCount = byteCount
        self.replicas = [Int : DeviceMemory]()
        self.isReadOnly = isReadOnly
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
            replicas[device]!.deallocate()
        } else {
            replicas.values.forEach { $0.deallocate() }
        }
    }
}

//==============================================================================
/// MemoryManagement
public extension MemoryManagement where Self: PlatformService {
    //--------------------------------------------------------------------------
    // bufferName
    func bufferName(_ ref: BufferRef) -> String {
        assert(deviceBuffers[ref.id] != nil, "Invalid BufferRef")
        return deviceBuffers[ref.id]!.name
    }
    
    //--------------------------------------------------------------------------
    // nextBufferRef
    var nextBufferRef: BufferRef { Platform.nextBufferRef }
    
    //--------------------------------------------------------------------------
    // createBuffer
    func createBuffer<Element>(of type: Element.Type, count: Int, name: String)
        -> BufferRef
    {
        let ref = self.nextBufferRef
        let byteCount = count * MemoryLayout<Element>.size
        deviceBuffers[ref.id] = DeviceBuffer(byteCount: byteCount, name: name)
        return ref
    }
    
    //--------------------------------------------------------------------------
    // createBuffer
    func createBuffer<Shape, Stream>(block shape: Shape, bufferedBlocks: Int,
                                     stream: Stream) -> (BufferRef, Int)
        where Shape : ShapeProtocol, Stream : BufferStream
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // cachedBuffer
    func cachedBuffer<Element>(for element: Element) -> BufferRef
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // createReference
    // create the DeviceBuffer record and add it to the dictionary
    func createReference<Element>(to buffer: UnsafeBufferPointer<Element>,
                                  name: String) -> BufferRef
    {
        // get a reference id
        let ref = self.nextBufferRef
        
        // create a device buffer entry for the id
        let roBuffer = UnsafeRawBufferPointer(buffer)
        let pointer = UnsafeMutableRawPointer(mutating: roBuffer.baseAddress!)
        let rawBuffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                      count: roBuffer.count)
        var deviceBuffer = DeviceBuffer(byteCount: rawBuffer.count,
                                        name: name, isReadOnly: true)
        deviceBuffer.replicas[0] = DeviceMemory(buffer: rawBuffer,
                                                    addressing: .unified, { })
        deviceBuffers[ref.id] = deviceBuffer
        return ref
    }
    
    //--------------------------------------------------------------------------
    // createMutableReference
    // create the DeviceBuffer record and add it to the dictionary
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferRef
    {
        // get a reference id
        let ref = self.nextBufferRef
        
        // create a device buffer entry for the id
        let rawBuffer = UnsafeMutableRawBufferPointer(buffer)
        var deviceBuffer = DeviceBuffer(byteCount: rawBuffer.count,
                                        name: name, isReadOnly: false)
        deviceBuffer.replicas[0] = DeviceMemory(buffer: rawBuffer,
                                                    addressing: .unified, { })
        deviceBuffers[ref.id] = deviceBuffer
        return ref
    }
    
    //--------------------------------------------------------------------------
    // duplicate
    func duplicate(_ other: BufferRef, using queue: QueueId) -> BufferRef {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // release
    func release(_ ref: BufferRef) {
        deviceBuffers[ref.id]!.deallocate()
    }
    
    //--------------------------------------------------------------------------
    // read
    func read<Element>(_ ref: BufferRef, of type: Element.Type,
                       at offset: Int, count: Int, using queueId: QueueId)
        -> UnsafeBufferPointer<Element>
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    func readWrite<Element>(_ ref: BufferRef, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queueId: QueueId)
        -> UnsafeMutableBufferPointer<Element>
    {
        // record the mutating queueId
        deviceBuffers[ref.id]!.lastMutatingQueue = queueId
        
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    /// migrate
    /// Migrates the master version of the data from wherever it is to
    /// the device associated with `queue` and returns a pointer to the data
    func migrate<Element>(_ ref: BufferRef, of type: Element.Type,
                          readOnly: Bool, using queueId: QueueId) throws
        -> DeviceMemory
    {
        // get a reference to the device buffer
        let device = queueId.device
        var buffer = deviceBuffers[ref.id]!
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
            buffer.masterDevice = device
            buffer.masterVersion += 1
        }
        deviceMemory.version = buffer.masterVersion
        
        // update collection
        deviceBuffers[ref.id]!.replicas[device] = deviceMemory
        return deviceMemory
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
        if let deviceMemory = buffer.replicas[device] {
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
