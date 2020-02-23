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
/// MemoryManager
public final class MemoryManager: MemoryManagement {
    public var deviceBuffers: [[Int : BufferDescription]]
    public var masterVersion: [Int : Int]
    // nextBufferId
    static var _nextBufferId: Int = 0
    public var nextBufferId: BufferId {
        Self._nextBufferId += 1
        return BufferId(Self._nextBufferId)
    }
    
    @inlinable
    public init() {
        deviceBuffers = [[Int : BufferDescription]]()
        masterVersion = [Int : Int]()
    }
}

//==============================================================================
/// BufferDescription
public struct BufferDescription {
    /// pointer to device buffer
    var buffer: UnsafeMutableRawBufferPointer
    /// `true` if the buffer can be mutated. The type of `buffer` is
    /// defined as mutable, but covers both cases to reduce generic complexity.
    let isMutable: Bool
    /// a buffer name used in diagnostic messages
    let name: String
    /// the mutation version of the buffer used for synchronization
    var version: Int
}

//==============================================================================
/// BufferId
/// a reference counted id for a service device buffer
public class BufferId: Equatable {
    public let id: Int

    @inlinable
    public init(_ id: Int) { self.id = id }

    /// a buffer name used in diagnostic messages
    @inlinable
    public var name: String { Platform.memory.bufferName(self) }

    public static func == (lhs: BufferId, rhs: BufferId) -> Bool {
        lhs.id == rhs.id
    }
}

//==============================================================================
/// DeviceStreamReader
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }

}

//==============================================================================
/// MemoryManagement
public protocol MemoryManagement: class {
    /// a collection of device buffer dictionaries indexed by the device
    /// number, and keyed by the id returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [[Int : BufferDescription]] { get set }
    
    /// a dictionary relating a buffer id to which device has the
    /// most recently mutated version. This is updated each time a write
    /// buffer is obtained on a different device
    /// - Parameter key: the buffer id
    /// - Parameter value: the index of the device that has the master version
    var masterVersion: [Int : Int] { get set }
    
    /// generates a unique buffer id
    var nextBufferId: BufferId { get }
    
    //--------------------------------------------------------------------------
    init()
    
    //--------------------------------------------------------------------------
    /// `bufferName(id:`
    /// - Parameter buffer: the id of the buffer
    /// - Returns: the name of the buffer used in diagnostic messages
    func bufferName(_ buffer: BufferId) -> String
    
    /// `createBuffer(type:count:`
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// A `BufferId` is used so the associated memory can be moved by the
    /// service between accesses in order to maximize memory utilization.
    /// - Parameter type: the element type of the buffer
    /// - Parameter count: size of the buffer in `Element` units
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createBuffer<Element>(of type: Element.Type, count: Int,
                               name: String) -> BufferId

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
                                     stream: Stream) -> (BufferId, Int)
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
    /// A BufferId is created if it does not already exist.
    func cachedBuffer<Element>(for element: Element) -> BufferId

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
        name: String) -> BufferId

    /// `createMutableReference(buffer:`
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferId

    /// `duplicate(other:queue:`
    /// makes a duplicate of the specified device buffer. Used to support
    /// copy-on-write semantics
    /// - Parameter other: the id of the other device buffer to duplicate
    /// - Parameter queue: specifies the device/queue for synchronization.
    /// - Returns: a reference to the device buffer
    func duplicate(_ other: BufferId, using queue: QueueId) -> BufferId

    /// `release(buffer:`
    /// Releases a buffer created by calling `createBuffer`
    /// - Parameter buffer: the device buffer to release
    func release(_ buffer: BufferId)

    /// `read(buffer:type:offset:queue:`
    /// - Parameter buffer: the device buffer id to read
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queue: device queue for data placement and synchronization
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized
    func read<Element>(_ buffer: BufferId, of type: Element.Type,
                       at offset: Int, count: Int,
                       using queue: DeviceQueue) -> UnsafeBufferPointer<Element>

    /// `readWrite(buffer:type:offset:queue:willOverwrite:`
    /// - Parameter buffer: the device buffer id to readWrite
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Parameter queue: device queue for data placement and synchronization
    /// - Returns: a mutable buffer pointer to the elements associated with the
    /// specified buffer id. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite<Element>(_ buffer: BufferId, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
/// MemoryManagement
public extension MemoryManagement {
    //--------------------------------------------------------------------------
    // bufferName
    func bufferName(_ buffer: BufferId) -> String {
        assert(deviceBuffers[0][buffer.id] != nil, "Invalid BufferId")
        return deviceBuffers[0][buffer.id]!.name
    }
    
    //--------------------------------------------------------------------------
    // createBuffer
    func createBuffer<Element>(of type: Element.Type, count: Int, name: String)
        -> BufferId
    {
        let bufferId = nextBufferId
        let rawCount = MemoryLayout<Element>.size * count
        let pointer = UnsafeMutableRawBufferPointer(start: nil, count: rawCount)
        let desc = BufferDescription(buffer: pointer, isMutable: true,
                                     name: name, version: 0)
        deviceBuffers[0][bufferId.id] = desc
        return bufferId
    }
    
    //--------------------------------------------------------------------------
    // createBuffer
    func createBuffer<Shape, Stream>(block shape: Shape, bufferedBlocks: Int,
                                     stream: Stream) -> (BufferId, Int)
        where Shape : ShapeProtocol, Stream : BufferStream
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // cachedBuffer
    func cachedBuffer<Element>(for element: Element) -> BufferId
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // createReference
    func createReference<Element>(to buffer: UnsafeBufferPointer<Element>,
                                  name: String) -> BufferId
    {
        let bufferId = nextBufferId
        let rawCount = MemoryLayout<Element>.size * buffer.count
        let rawpPointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        let rawBuffer = UnsafeMutableRawBufferPointer(start: rawpPointer,
                                                      count: rawCount)
        let desc = BufferDescription(buffer: rawBuffer, isMutable: true,
                                     name: name, version: 0)
        deviceBuffers[0][bufferId.id] = desc
        return bufferId
    }
    
    //--------------------------------------------------------------------------
    // createMutableReference
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferId
    {
        let bufferId = nextBufferId
        let rawCount = MemoryLayout<Element>.size * buffer.count
        let rawpPointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        let rawBuffer = UnsafeMutableRawBufferPointer(start: rawpPointer,
                                                      count: rawCount)
        let desc = BufferDescription(buffer: rawBuffer, isMutable: true,
                                     name: name, version: 0)
        deviceBuffers[0][bufferId.id] = desc
        return bufferId
    }
    
    //--------------------------------------------------------------------------
    // duplicate
    func duplicate(_ other: BufferId, using queue: QueueId) -> BufferId {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // release
    func release(_ bufferId: BufferId) {
    }
    
    //--------------------------------------------------------------------------
    // read
    func read<Element>(_ buffer: BufferId, of type: Element.Type,
                       at offset: Int, count: Int, using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    func readWrite<Element>(_ buffer: BufferId, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<Element>
    {
        fatalError()
    }
}
