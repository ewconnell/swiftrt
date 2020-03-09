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
    associatedtype Element
    
    /// the id of the buffer for diagnostics
    var id: Int { get }
    /// `true` if the buffer is read only
    var isReadOnly: Bool { get }
    /// `true` if this buffer is a reference to an application managed buffer
    var isReference: Bool { get }
    /// the buffer name used in diagnostic messages
    var name: String { get }
    
    /// `init(count:name:
    /// creates a lazily allocated element buffer
    /// - Parameter count: size of the buffer in `Element` units
    /// - Parameter name: name used in diagnostic messages
    init(count: Int, name: String)
    
    /// `init(copying other:`
    /// copy constructor
    init(copying other: Self)
    
    /// `init(elements:name:`
    /// creates a lazily allocated element buffer
    /// - Parameter elements: a collection of initial buffer elements
    /// - Parameter name: name used in diagnostic messages
    init<C>(elements: C, name: String)
        where C: Collection, C.Element == Element
    
    /// `init(buffer:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter buffer: a buffer pointer to the data
    /// - Parameter name: name used in diagnostic messages
    init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String)
    
    /// `init(buffer:`
    /// creates an element buffer whose data is managed by the application.
    /// No memory is allocated, so the buffer must point to valid data space.
    /// - Parameter buffer: a mutable buffer pointer to application data
    /// - Parameter name: name used in diagnostic messages
    init(referenceTo buffer: UnsafeMutableBufferPointer<Element>, name: String)
    
    /// `init(blockSize:bufferedBlocks:sequence:`
    /// initializes a streaming device buffer to be used with `stream`
    /// - Parameter type: the element type of the buffer
    /// - Parameter shape: the shape of the blocks read or written to
    /// the sequence in a given transaction. This might be the number
    /// of elements in a view.
    /// - Parameter bufferedBlocks: the size of the device buffer
    /// to reserve in block units
    /// - Parameter stream: the I/O object for read/write operations
    init<B, Stream>(block shape: Shape<B>, bufferedBlocks: Int, stream: Stream)
        where B: ShapeBounds, Stream: BufferStream
    
    /// `init(element:name:
    /// initializes an element buffer for the specified `Element` value.
    /// User expressions use a lot of constant scalar values
    /// which are repeated. For example: `let m = matrix + 1`. These
    /// expressions are frequently iterated thousands of times. This initializer
    /// can access a cache of constant value buffers, which are likely to
    /// already be present on a discreet accelerator device.
    /// - Parameter element: the element value
    init(for element: Element, name: String)
    
    /// `read(offset:count:
    /// gets a buffer pointer blocking the calling thread until synchronized
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    /// when the queue reaches this point
    func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element>
    
    /// `read(offset:count:queue:`
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queue: queue for device placement and synchronization
    /// - Returns: a buffer pointer to the elements. Elements will be valid
    /// when the queue reaches this point
    func read(at offset: Int, count: Int, using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    
    /// `readWrite(type:offset:count:willOverwrite:
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements.
    /// Elements will be valid when the queue reaches this point
    func readWrite(at offset: Int, count: Int, willOverwrite: Bool)
        -> UnsafeMutableBufferPointer<Element>

    /// `readWrite(type:offset:count:willOverwrite:queue:
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter count: the number of elements to be accessed
    /// - Parameter queue: queue for device placement and synchronization
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements.
    /// Elements will be valid when the queue reaches this point
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

//==============================================================================
/// ShapedBuffer
public protocol ShapedBuffer: Collection {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var pointer: UnsafeBufferPointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// BufferElements
public struct BufferElements<Element, Shape>: ShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Shape.Index
    public let pointer: UnsafeBufferPointer<Element>
    public let shape: Shape

    @inlinable public var endIndex: Index { shape.endIndex }
    @inlinable public var startIndex: Index { shape.startIndex }

    //-----------------------------------
    // initializers
    @inlinable
    public init(_ shape: Shape, _ pointer: UnsafeBufferPointer<Element>) {
        assert(pointer.count > 0, "can't enumerate an empty shape")
        self.shape = shape
        self.pointer = pointer
    }
    
    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index { shape.index(after: i) }

    @inlinable
    public subscript(index: Index) -> Element {
        pointer[shape[index]]
    }
}

//==============================================================================
/// MutableShapedBuffer
public protocol MutableShapedBuffer: MutableCollection {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var pointer: UnsafeMutableBufferPointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// MutableBufferElements
public struct MutableBufferElements<Element, Shape>: MutableShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Shape.Index
    public let pointer: UnsafeMutableBufferPointer<Element>
    public let shape: Shape
    
    @inlinable public var endIndex: Index { shape.endIndex }
    @inlinable public var startIndex: Index { shape.startIndex }
    
    //-----------------------------------
    // initializers
    @inlinable
    public init(_ shape: Shape, _ pointer: UnsafeMutableBufferPointer<Element>)
    {
        assert(pointer.count > 0, "can't enumerate an empty shape")
        self.shape = shape
        self.pointer = pointer
    }
    
    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index { shape.index(after: i) }

    @inlinable
    public subscript(index: Index) -> Element {
        get {
            pointer[shape[index]]
        }
        set {
            pointer[shape[index]] = newValue
        }
    }
}

