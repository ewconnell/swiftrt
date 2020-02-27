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
/// ElementBuffer protocol
public protocol ElementBuffer: class {
    /// the id of the buffer for diagnostics
    var id: Int { get }
    /// `true` if the buffer is read only
    var isReadOnly: Bool { get }
    /// `true` if this buffer is a reference to an application managed buffer
    var isReference: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get }

    /// `duplicate(`
    /// - Returns:a copy of the buffer
    func duplicate() -> ElementBuffer
    
    /// `read(type:offset:count:queue:`
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    /// when the queue reaches this point
    func read<E>(type: E.Type, at offset: Int, count: Int,
                 using queue: DeviceQueue) -> UnsafeBufferPointer<E>
    
    /// `readWrite(type:offset:count:willOverwrite:queue:
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queue: queue for device placement and synchronization
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements.
    /// Elements will be valid when the queue reaches this point
    func readWrite<E>(type: E.Type, at offset: Int, count: Int,
                      willOverwrite: Bool, using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<E>
}

//==============================================================================
/// MemoryManagement
public protocol MemoryManagement: class {
    /// `createBuffer(type:count:name:
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// - Parameter type: the element type of the buffer
    /// - Parameter count: size of the buffer in `Element` units
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: an element buffer reference
    func createBuffer<E>(of type: E.Type, count: Int, name: String)
        -> BufferRef
    
    /// `createBuffer(blockSize:bufferedBlocks:sequence:`
    /// creates a streaming device buffer to be used in tensor operations.
    /// - Parameter type: the element type of the buffer
    /// - Parameter shape: the shape of the blocks read or written to
    /// the sequence in a given transaction. This might be the number
    /// of elements in a view.
    /// - Parameter bufferedBlocks: the size of the device buffer
    /// to reserve in block units
    /// - Parameter stream: the I/O object for read/write operations
    /// - Returns: a buffer id and the size of the stream in block units.
    /// An endless sequence will return infinity for the block count.
    func createBuffer<E, Shape, Stream>(
        of type: E.Type, block shape: Shape,
        bufferedBlocks: Int, stream: Stream) -> (BufferRef, Int)
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
    /// - Returns: an element buffer that contains the element value.
    /// A buffer is created if it does not already exist.
    func cachedBuffer<E>(for element: E) -> BufferRef
    
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
    func createReference<E>(to buffer: UnsafeBufferPointer<E>,
                            name: String) -> BufferRef
    
    /// `createMutableReference(buffer:`
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Parameter name: name used in diagnostic messages
    /// - Returns: a reference to the device buffer
    func createMutableReference<E>(to buffer: UnsafeMutableBufferPointer<E>,
                                   name: String) -> BufferRef
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }
    
}

