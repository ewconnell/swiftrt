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
    associatedtype Element: StorageElement

    /// the number of storage elements
    var count: Int { get }
    /// the id of the buffer for diagnostics
    var id: Int { get }
    /// `true` if the buffer is read only
    var isReadOnly: Bool { get }
    /// `true` if this buffer is a reference to an application managed buffer
    var isReference: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get set }
    /// specifies the stored element layout order
    var layout: Layout { get }
    
    /// `init(count:layout:
    /// creates an uninitialized lazily allocated element buffer
    /// - Parameters:
    ///  - count: size of the buffer in `Element` units
    ///  - layout: element layout order
    init(count: Int, layout: Layout)
    
    /// `init(element:
    /// creates a storage buffer with a single element
    /// - Parameters:
    ///  - element: the initial element value
    init(single element: Element.Value)
    
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    /// `init(buffer:layout:
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameters:
    ///  - buffer: a buffer pointer to the data
    ///  - layout: element layout order
    init(referenceTo buffer: UnsafeBufferPointer<Element.Stored>,
         layout: Layout)
    
    /// `init(buffer:layout:
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameters:
    ///  - buffer: a mutable buffer pointer to application data
    ///  - layout: element layout order
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element.Stored>,
         layout: Layout)
    
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
    /// - Parameter index: the linear storage index of the element
    /// - Returns: a single element at the specified offset
    func element(at index: Int) -> Element.Value
    
    /// `setElement(value:offset:`
    /// - Parameters:
    ///  - value: the value to set
    ///  - index: the linear storage index of the element
    func setElement(value: Element.Value, at index: Int)
    
    /// `read(offset:count:`
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameters:
    ///  - offset: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element.Stored>
    
    /// `read(index:count:queue:`
    /// - Parameters:
    ///  - index: the buffer base index within storage
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at index: Int, count: Int,
              using queue: PlatformType.Device.Queue)
        -> UnsafeBufferPointer<Element.Stored>
    
    /// `readWrite(type:index:count:willOverwrite:
    /// - Parameters:
    ///  - index: the buffer base offset within storage
    ///  - count: the number of elements to be accessed
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite(at index: Int, count: Int)
        -> UnsafeMutableBufferPointer<Element.Stored>

    /// `readWrite(type:index:count:queue:
    /// - Parameters:
    ///  - index: the buffer base index within storage
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite(at index: Int, count: Int,
                   using queue: PlatformType.Device.Queue)
        -> UnsafeMutableBufferPointer<Element.Stored>
}

public extension StorageBuffer {
    @inlinable var diagnosticName: String { "\(name)(\(id))" }
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }
    
}
