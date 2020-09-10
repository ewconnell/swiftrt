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
    /// `true` if the tensor value is zero.
    // Note: This is used to minimize the AD zero materialization design problem
    var isZero: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get set }
    
    //--------------------------------------------------------------------------
    /// `init(type:count:
    /// creates an uninitialized lazily allocated buffer to hold `count`
    /// number of `Element`s
    /// - Parameters:
    ///  - storedType: the type of storage `Element`
    ///    This is used to compute buffer byte size and alignment.
    ///  - storedCount: the number of `Element`s stored in the buffer.
    ///  - name: the name of the tensor
    init<Element>(
        storedType: Element.Type,
        count: Int,
        name: String
    )
    
    //--------------------------------------------------------------------------
    /// `init(single:name:
    /// creates storage for a single element value
    /// - Parameters:
    ///  - storedElement: the stored element
    ///  - name: the name of the tensor
    init<Element>(
        storedElement: Element,
        name: String
    )
    
    //--------------------------------------------------------------------------
    /// `init(type:other:queue:`
    /// creates a copy of the storage using `Context.currentQueue`
    /// - Parameters:
    ///  - type: the type of element to copy
    ///  - other: the storage to copy
    ///  - queue: the device queue to use
    init<Element>(
        type: Element.Type,
        copying other: Self,
        using queue: DeviceQueue
    )
    
    //--------------------------------------------------------------------------
    /// `init(buffer:name:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameters:
    ///  - buffer: the referenced `Element` buffer
    ///  - name: the name of the tensor
    init<Element>(referenceTo buffer: UnsafeBufferPointer<Element>,
                  name: String)
    
    //--------------------------------------------------------------------------
    /// `init(buffer:order:name:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameters:
    ///  - buffer: the referenced `Element` buffer
    ///  - name: the name of the tensor
    init<Element>(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                  name: String)
    
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
        using queue: DeviceQueue
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
        using queue: DeviceQueue
    ) -> UnsafeMutableBufferPointer<Element>
    
    //--------------------------------------------------------------------------
    /// waitForCompletion
    /// blocks the caller until pending write operations have completed
    func waitForCompletion()
}

//==============================================================================
// convenience extensions
//
public extension StorageBuffer {
    /// used for unit tests. `true` if a read/write operation caused
    /// memory to be copied between devices
    @inlinable var testLastAccessCopiedDeviceMemory: Bool { false }
    
    //--------------------------------------------------------------------------
    /// `init(type:count:order:
    /// creates an uninitialized lazily allocated buffer to hold `count`
    /// number of tensor `Element`s
    /// - Parameters:
    ///  - type: the type of tensor `Element`
    ///    This is used to compute buffer byte size and alignment.
    ///  - count: the number of `Element`s stored in the buffer.
    ///  - name: the name of the tensor
    @inlinable init<Element: StorageElement>(
        type: Element.Type,
        count: Int,
        name: String
    ) {
        self.init(storedType: Element.Stored.self,
                  count: Element.storedCount(count),
                  name: name)
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
    /// waitForCompletion
    /// blocks the caller until pending write operations have completed
    @inlinable func waitForCompletion() { }
}

//==============================================================================
/// BufferStream
/// a reference counted stream reader
public protocol BufferStream {
    /// `true` if the stream can be written to
    var isMutable: Bool { get }
    
}
