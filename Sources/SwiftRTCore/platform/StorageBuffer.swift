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
    
    //--------------------------------------------------------------------------
    /// `init(type:count:layout:
    /// creates an uninitialized lazily allocated buffer to hold `count`
    /// number of `Element`s
    /// - Parameters:
    ///  - storedType: the type of storage `Element`
    ///    This is used to compute buffer byte size and alignment.
    ///  - storedCount: the number of `Element`s stored in the buffer.
    ///  - layout: element memory layout order
    ///  - name: the name of the tensor
    init<Element>(
        storedType: Element.Type,
        count: Int,
        layout: Layout,
        name: String
    )
    
    //--------------------------------------------------------------------------
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    //--------------------------------------------------------------------------
    /// `init(buffer:layout:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameters:
    ///  - buffer: the referenced `Element` buffer
    ///  - layout: element layout order
    init<Element>(
        referenceTo buffer: UnsafeBufferPointer<Element>,
        layout: Layout
    )
    
    //--------------------------------------------------------------------------
    /// `init(buffer:layout:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameters:
    ///  - buffer: the referenced `Element` buffer
    ///  - layout: element layout order
    init<Element>(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>,
        layout: Layout
    )
    
    //--------------------------------------------------------------------------
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
        
    //--------------------------------------------------------------------------
    /// `read(type:index:count:queue:`
    /// - Parameters:
    ///  - type: the element type to bind
    ///  - index: the element index where the returned buffer will start
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: an element buffer pointer. The caller
    ///   is required to do appropriate index transformations for packed
    ///   element types. Any required memory transfers are added to the
    ///   specified queue.
    func read<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeBufferPointer<Element>
    
    //--------------------------------------------------------------------------
    /// `readWrite(type:index:count:queue:`
    /// - Parameters:
    ///  - type: the element type to bind
    ///  - index: the element index where the returned buffer will start
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a mutable buffer pointer to elements. The caller
    ///   is required to do appropriate index transformations for packed
    ///   element types. Any required memory transfers are added to the
    ///   specified queue.
    func readWrite<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
// convenience extensions
//
public extension StorageBuffer {
    @inlinable var diagnosticName: String { "\(name)(\(id))" }
    
    //--------------------------------------------------------------------------
    /// `init(type:count:layout:
    /// creates an uninitialized lazily allocated buffer to hold `count`
    /// number of tensor `Element`s
    /// - Parameters:
    ///  - type: the type of tensor `Element`
    ///    This is used to compute buffer byte size and alignment.
    ///  - count: the number of `Element`s stored in the buffer.
    ///  - layout: element memory layout order
    ///  - name: the name of the tensor
    @inlinable init<Element: StorageElement>(
        type: Element.Type,
        count: Int,
        layout: Layout,
        name: String = "Tensor"
    ) {
        self.init(storedType: Element.Stored.self,
                  count: Element.storedCount(count),
                  layout: layout, name: name)
    }

    //--------------------------------------------------------------------------
    /// countOf(type:
    /// - Returns: the number of `Element`s in the storage
    @inlinable func countOf<Element>(type: Element.Type) -> Int {
        assert(byteCount % MemoryLayout<Element>.size == 0,
               "Buffer size is not even multiple of Element type")
        return byteCount / MemoryLayout<Element>.size
    }

    //--------------------------------------------------------------------------
    /// `element(type:at:`
    /// - Parameters:
    ///  - type: the type of tensor `Element` (e.g. Float, UInt8, etc..)
    ///  - index: the absolute logical linear storage index of the element
    /// - Returns: a single element at the specified offset
    @inlinable func element<E: StorageElement>(
        type: E.Type,
        at index: Int
    ) -> E.Value {
        let i = E.storedIndex(index)
        let buffer = read(type: E.Stored.self, at: i, count: 1)
        return E.value(at: index, from: buffer[0])
    }
    
    //--------------------------------------------------------------------------
    /// `setElement(type:value:offset:`
    /// - Parameters:
    ///  - type: the type of tensor `Element` (e.g. Float, UInt8, etc..)
    ///  - value: the value to set
    ///  - index: the absolute logical linear storage index of the element
    @inlinable func setElement<E: StorageElement>(
        type: E.Type,
        value: E.Value,
        at index: Int
    ) {
        let i = E.storedIndex(index)
        let mutableBuffer = readWrite(type: E.Stored.self, at: i, count: 1)
        E.store(value: value, at: index, to: &mutableBuffer[0])
    }
    
    //--------------------------------------------------------------------------
    /// `read(type:index:count:`
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameters:
    ///  - type: the type of storage element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    @inlinable func read<Element>(
        type: Element.Type,
        at base: Int,
        count: Int
    ) -> UnsafeBufferPointer<Element> {
        read(type: type, at: base, count: count, using: Context.cpuQueue(0))
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite(type:index:count`
    /// - Parameters:
    ///  - type: the type of storage element
    ///  - base: the base storage index of the returned buffer
    ///  - count: the number of elements to be accessed
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    @inlinable func readWrite<Element>(
        type: Element.Type,
        at base: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<Element> {
        readWrite(type: type, at: base, count: count, using: Context.cpuQueue(0))
    }
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }
    
}
