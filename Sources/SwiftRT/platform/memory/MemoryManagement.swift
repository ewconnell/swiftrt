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
    /// - Parameter ref: the id of the other device buffer to duplicate
    /// - Parameter queue: specifies the device/queue for synchronization.
    /// - Returns: a reference to the device buffer
    func duplicate(_ ref: BufferRef, using queue: QueueId) -> BufferRef

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
/// MemoryManagement extensions
public extension MemoryManagement {
    // nextBufferRef
    var nextBufferRef: BufferRef { Platform.nextBufferRef }
}

//==============================================================================
/// BufferRef
/// a class object used to maintain a reference count to a set of
/// associated device buffers
public class BufferRef: Equatable {
    /// used to identify the `DeviceBuffer` instance
    public let id: Int

    // initializer
    @inlinable public init(_ id: Int) { self.id = id }

    // release the associated buffer
    deinit {
        Platform.service.release(self)
    }
    
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

