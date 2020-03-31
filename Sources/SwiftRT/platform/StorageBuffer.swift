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

    /// the buffer used for host access
    var hostBuffer: UnsafeMutableBufferPointer<Element> { get }
    /// the id of the buffer for diagnostics
    var id: Int { get }
    /// `true` if the buffer is read only
    var isReadOnly: Bool { get }
    /// `true` if this buffer is a reference to an application managed buffer
    var isReference: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get set }
    
    /// `init(count:name:
    /// creates a lazily allocated element buffer
    /// - Parameters:
    ///  - count: size of the buffer in `Element` units
    ///  - name: name used in diagnostic messages
    ///  - value: optional initial element value
    init(count: Int, name: String, element value: Element?)
    
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    /// `init(elements:name:`
    /// creates a lazily allocated element buffer
    /// - Parameters:
    ///  - elements: a collection of initial buffer elements
    ///  - name: name used in diagnostic messages
    init<C>(elements: C, name: String)
        where C: Collection, C.Element == Element
    
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
    
    /// `init(element:name:
    /// initializes an element buffer for the specified `Element` value.
    /// User expressions use a lot of constant scalar values
    /// which are repeated. For example: `let m = matrix + 1`. These
    /// expressions are frequently iterated thousands of times. This initializer
    /// can access a cache of constant value buffers, which are likely to
    /// already be present on a discreet accelerator device.
    /// - Parameters:
    ///  - element: the element value
    ///  - name: name used in diagnostic messages
    init(for element: Element, name: String)
    
    /// `read(offset:count:
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameters:
    ///  - offset: the offset in element sized units from
    ///    the beginning of the buffer to read
    ///  - count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element>
    
    /// `read(offset:count:queue:`
    /// - Parameters:
    ///  - offset: the offset in element sized units from
    ///    the beginning of the buffer to read
    ///  - count: the number of elements to be accessed
    ///  - queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    ///   when the queue reaches this point
    func read(at offset: Int, count: Int, using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    
    /// `readWrite(type:offset:count:willOverwrite:
    /// - Parameters:
    ///  - offset: the offset in element sized units from
    ///    the beginning of the buffer to read
    ///  - count: the number of elements to be accessed
    ///  - willOverwrite: `true` if the caller guarantees all
    ///    buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements.
    ///   Elements will be valid when the queue reaches this point
    func readWrite(at offset: Int, count: Int, willOverwrite: Bool)
        -> UnsafeMutableBufferPointer<Element>

    /// `readWrite(type:offset:count:willOverwrite:queue:
    /// - Parameters:
    ///  - offset: the offset in element sized units from
    ///    the beginning of the buffer to read
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
//
////==============================================================================
///// ShapedBuffer
//public protocol ShapedBuffer: Collection {
//    associatedtype Element
//    associatedtype Bounds: ShapeBounds
//
//    var pointer: UnsafeBufferPointer<Element> { get }
//    var shape: TensorShape<Bounds> { get }
//}
//
////==============================================================================
///// BufferElements
//public struct BufferElements<Element, Bounds>: ShapedBuffer
//    where Bounds: ShapeBounds
//{
//    public typealias Index = TensorShape<Bounds>.Index
//    public let pointer: UnsafeBufferPointer<Element>
//    public let shape: TensorShape<Bounds>
//
//    @inlinable public var endIndex: Index { shape.endIndex }
//    @inlinable public var startIndex: Index { shape.startIndex }
//
//    //-----------------------------------
//    // initializers
//    @inlinable
//    public init(_ shape: TensorShape<Bounds>,
//                _ pointer: UnsafeBufferPointer<Element>)
//    {
//        assert(pointer.count > 0, "can't enumerate an empty shape")
//        self.shape = shape
//        self.pointer = pointer
//    }
//
//    //-----------------------------------
//    // Collection
//    @inlinable
//    public func index(after i: Index) -> Index { shape.index(after: i) }
//
//    @inlinable
//    public subscript(index: Index) -> Element {
//        pointer[shape[index]]
//    }
//}
//
////==============================================================================
///// MutableShapedBuffer
//public protocol MutableShapedBuffer: MutableCollection {
//    associatedtype Element
//    associatedtype Bounds: ShapeBounds
//
//    var pointer: UnsafeMutableBufferPointer<Element> { get }
//    var shape: TensorShape<Bounds> { get }
//}
//
////==============================================================================
///// MutableBufferElements
//public struct MutableBufferElements<Element, Bounds>: MutableShapedBuffer
//    where Bounds: ShapeBounds
//{
//    public typealias Index = TensorShape<Bounds>.Index
//    public let pointer: UnsafeMutableBufferPointer<Element>
//    public let shape: TensorShape<Bounds>
//
//    @inlinable public var endIndex: Index { shape.endIndex }
//    @inlinable public var startIndex: Index { shape.startIndex }
//
//    //-----------------------------------
//    // initializers
//    @inlinable
//    public init(_ shape: TensorShape<Bounds>,
//                _ pointer: UnsafeMutableBufferPointer<Element>)
//    {
//        assert(pointer.count > 0, "can't enumerate an empty shape")
//        self.shape = shape
//        self.pointer = pointer
//    }
//
//    //-----------------------------------
//    // Collection
//    @inlinable
//    public func index(after i: Index) -> Index { shape.index(after: i) }
//
//    @inlinable
//    public subscript(index: Index) -> Element {
//        get {
//            pointer[shape[index]]
//        }
//        set {
//            pointer[shape[index]] = newValue
//        }
//    }
//}
//
