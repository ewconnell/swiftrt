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
/// StorageBuffer protocol
public protocol StorageBuffer: class, Logging {
    /// the type of element stored in the buffer
    associatedtype Element

    /// the id of the buffer for diagnostics
    var id: Int { get }
    /// `true` if the buffer is read only
    var isReadOnly: Bool { get }
    /// `true` if this buffer is a reference to an application managed buffer
    var isReference: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get set }
    
    /// `init(count:name:
    /// creates an uninitialized lazily allocated element buffer
    /// - Parameters:
    ///  - count: size of the buffer in `Element` units
    ///  - name: name used in diagnostic messages
    init(count: Int, name: String)
    
    /// `init(element:name:
    /// creates a storage buffer with a single element
    /// - Parameters:
    ///  - element: the initial element value
    ///  - name: name used in diagnostic messages
    init(single element: Element, name: String)
    
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    /// `init(buffer:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameters:
    ///  - buffer: a buffer pointer to the data
    ///  - name: name used in diagnostic messages
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String)
    
    /// `init(buffer:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameters:
    ///  - buffer: a mutable buffer pointer to application data
    ///  - name: name used in diagnostic messages
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>, name: String)
    
    /// `init(blockSize:bufferedBlocks:sequence:`
    /// initializes a streaming device buffer to be used with `stream`
    /// - Parameters:
    ///  - type: the element type of the buffer
    ///  - shape: the shape of the blocks read or written to
    ///    the sequence in a given transaction. This might be the number
    ///    of elements in a view.
    ///  - bufferedBlocks: the size of the device buffer
    ///    to reserve in block units
    ///  - stream: the I/O object for read/write operations
    init<S, Stream>(block shape: S, bufferedBlocks: Int, stream: Stream)
        where S: TensorShape, Stream: BufferStream
        
    /// `element(offset:`
    /// - Parameter offset: the linear storage index of the element
    /// - Returns: a single element at the specified offset
    func element(at offset: Int) -> Element
    
    /// `setElement(value:offset:`
    /// - Parameters:
    ///  - value: the value to set
    ///  - offset: the linear storage index of the element
    func setElement(value: Element, at offset: Int)
    
    /// `read(offset:count:`
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameters:
    ///  - offset: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element>
    
    /// `read(offset:count:queue:`
    /// - Parameters:
    ///  - offset: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at offset: Int, count: Int, using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    
    /// `readWrite(type:offset:count:willOverwrite:
    /// - Parameters:
    ///  - offset: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite(at offset: Int, count: Int)
        -> UnsafeMutableBufferPointer<Element>

    /// `readWrite(type:offset:count:willOverwrite:queue:
    /// - Parameters:
    ///  - offset: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    ///  - willOverwrite: `true` if the caller guarantees all
    ///    buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite(at offset: Int, count: Int,
                   willOverwrite: Bool, using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }
    
}
