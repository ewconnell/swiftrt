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
/// The storage buffer is a container for tensor elements. It uses raw buffers
/// and per function type binding, so zero copy casting can be done.
/// For example casting `RGBA<Float>` to `NHWC` to interact with Cuda,
/// or `UInt1` to `Bool1` to manipulate bit tensors in different ways.
public protocol StorageBuffer: class, Logging {
    /// memory buffer alignment
    var alignment: Int { get }
    /// the number of storage elements
    var byteCount: Int { get }
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
    
    /// countOf(type:
    /// - Returns: the number of `type` units in the storage
    func countOf<E: StorageElement>(type: E) -> Int
    
    /// `init(type:count:layout:
    /// creates an uninitialized lazily allocated element buffer
    /// - Parameters:
    ///  - type: the type of storage element. Used to compute byte size
    ///  - count: size of the buffer in `Element` units
    ///  - layout: element layout order
    init<E: StorageElement>(type: E.Type, count: Int, layout: Layout)
    
    /// `init(element:
    /// creates a storage buffer with a single element
    /// - Parameters:
    ///  - type: the type of storage element
    ///  - element: the initial element value
    init<E: StorageElement>(type: E.Type, single element: E.Value)
    
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    /// `init(buffer:layout:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameters:
    ///  - type: the type of storage element
    ///  - buffer: a buffer pointer to the data
    ///  - layout: element layout order
    init<E: StorageElement>(
        type: E.Type,
        referenceTo buffer: UnsafeBufferPointer<E.Stored>,
        layout: Layout
    )
    
    /// `init(buffer:layout:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameters:
    ///  - type: the type of storage element
    ///  - buffer: a mutable buffer pointer to application data
    ///  - layout: element layout order
    init<E: StorageElement>(
        type: E.Type,
        referenceTo buffer: UnsafeMutableBufferPointer<E.Stored>,
        layout: Layout
    )
    
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
        
    /// `element(type:at:`
    /// - Parameters:
    ///  - type: the type of element
    ///  - index: the absolute linear storage index of the element
    /// - Returns: a single element at the specified offset
    func element<E: StorageElement>(
        type: E.Type,
        at index: Int
    ) -> E.Value
    
    /// `setElement(value:offset:`
    /// - Parameters:
    ///  - type: the type of element
    ///  - value: the value to set
    ///  - index: the absolute linear storage index of the element
    func setElement<E: StorageElement>(
        type: E.Type,
        value: E.Value,
        at index: Int
    )
    
    /// `read(type:index:count:`
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameters:
    ///  - type: the type of element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int
    ) -> UnsafeBufferPointer<E.Stored>
    
    /// `read(index:count:queue:`
    /// - Parameters:
    ///  - type: the type of element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeBufferPointer<E.Stored>
    
    /// `readWrite(type:index:count`
    /// - Parameters:
    ///  - type: the type of element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<E.Stored>

    /// `readWrite(type:index:count:queue:`
    /// - Parameters:
    ///  - type: the type of element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableBufferPointer<E.Stored>
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
