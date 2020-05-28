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
import Numerics

//==============================================================================
/// StorageElement
/// tensor elements conform to `StorageElement`, which enables reading, writing,
/// and operating on data types that are not native to the Swift language,
/// but can be native and optimal for gpus
public protocol StorageElement {
    associatedtype Stored
    associatedtype Value
    
    /// storedIndex
    /// the stored index will be less than the logical index for packed
    /// bit types such as `Int4`
    /// - Parameter index: the logical buffer index
    /// - Returns: the stored index
    static func storedIndex(_ index: Int) -> Int
    
    /// storedCount
    /// the stored count will be less than the logical count for packed
    /// bit types such as `Int4`
    /// - Parameter count: the logical buffer count
    /// - Returns: the stored count
    static func storedCount(_ count: Int) -> Int
    
    /// value
    /// converts from `Stored` type to `Value` type
    /// - Parameters:
    ///  - stored: the stored representation of the element
    ///  - index: the logical buffer index for the element. This is used
    ///    to calculate the subelement position for packed elements
    /// - Returns: the element interchange Value. For example, an `Int1`
    ///   element is interchanged with the caller as an `Int`
    static func value(from stored: Stored, at index: Int) -> Value
    
    /// store
    /// writes a value into a storage location
    /// - Parameters:
    ///  - value: the element value to store. For package element types,
    ///    it is an error if the interchange Value exceeds the
    ///    range of the packed type.
    ///  - index: the logical buffer index for the element. This is used
    ///    to calculate the subelement position for packed elements
    ///  - stored: a reference to the storage location. Packed elements
    ///    are combined (or'ed) when written.
    static func store(value: Value, at index: Int, to stored: inout Stored)

    /// stored
    /// converts the Value interchange type to the Stored type
    /// - Parameters:
    ///  - value: the element value to store
    static func stored(value: Value) -> Stored
}

//==============================================================================
// the default behavior for whole elements is simply pass through
public extension StorageElement {
    @inlinable static func storedIndex(_ index: Int) -> Int { index }
    @inlinable static func storedCount(_ count: Int) -> Int { count }
}

//-------------------------------------
// Stored == Value
public extension StorageElement where Stored == Value {
    @inlinable static func value(
        from stored: Stored,
        at index: Int
    ) -> Value { stored }
    
    @inlinable static func store(
        value: Value,
        at index: Int,
        to stored: inout Stored
    ) { stored = value }

    static func stored(value: Value) -> Stored { value }
}

//==============================================================================
public protocol PackedStorageElement: StorageElement {
    static var indexShift: Int { get }
    static var maskingShift: Int { get }
    static var valueMask: Stored { get }
    static var valueMin: Value { get }
    static var valueMax: Value { get }
}

public extension PackedStorageElement
    where Value: BinaryInteger, Stored: BinaryInteger
{
    @inlinable static func packedShift(_ index: Int) -> Int {
        (index % (1 << indexShift)) << maskingShift
    }
    
    @inlinable static func storedIndex(_ index: Int) -> Int {
        index >> indexShift
    }
    
    @inlinable static func storedCount(_ count: Int) -> Int {
        var storedCount = count >> indexShift
        if storedCount << indexShift != count { storedCount += 1 }
        return storedCount
    }
    
    @inlinable static func value(from stored: Stored, at index: Int) -> Value {
        Value(stored >> packedShift(index) & valueMask)
    }
    
    @inlinable static func store(
        value: Value, at index: Int, to stored: inout Stored
    ) {
        assert(value >= valueMin && value <= valueMax)
        let shiftCount = packedShift(index)
        if shiftCount == 0 {
            // init top bits with 0
            stored = Stored(value)
        } else {
            stored |= Stored(value) << shiftCount
        }
    }
    
    @inlinable static func stored(value: Value) -> Stored { Stored(value) }
}

//==============================================================================
// packed bit types that automatically cast to a native type during iteration
public struct UInt1: PackedStorageElement {
    public typealias Stored = UInt8
    public typealias Value = Int
    @inlinable public static var indexShift: Int { 3 }
    @inlinable public static var maskingShift: Int { 0 }
    @inlinable public static var valueMask: Stored { 0x1 }
    @inlinable public static var valueMin: Value { 0 }
    @inlinable public static var valueMax: Value { 1 }
}

public struct UInt4: PackedStorageElement {
    public typealias Stored = UInt8
    public typealias Value = Int
    @inlinable public static var indexShift: Int { 1 }
    @inlinable public static var maskingShift: Int { 2 }
    @inlinable public static var valueMask: Stored { 0x0F }
    @inlinable public static var valueMin: Value { 0 }
    @inlinable public static var valueMax: Value { 15 }
}

//==============================================================================
// non native types that automatically cast to a native type during iteration
extension Float16: StorageElement {
    @inlinable public static func value(
        from stored: Self, at index: Int
    ) -> Float { Float(stored) }
    
    @inlinable public static func store(
        value: Float, at index: Int, to stored: inout Self
    ) { stored = Self(value) }

    @inlinable public static func stored(value: Float) -> Self { Self(value) }
}

//==============================================================================
// standard native type conformance
extension Bool: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Int8: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension UInt8: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Int16: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension UInt16: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Int32: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension UInt32: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Float: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Double: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension Complex: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

//==============================================================================
// storage element iterators
//==============================================================================

@inlinable public func haveSameStorageLayout<S,E0,E1>(
    _ a: Tensor<S,E0>,
    _ b: Tensor<S,E1>
) -> Bool {
    a.layout == b.layout && a.isBufferIterable && b.isBufferIterable
}

@inlinable public func haveSameStorageLayout<S,E0, E1, E2>(
    _ a: Tensor<S,E0>,
    _ b: Tensor<S,E1>,
    _ c: Tensor<S,E2>
) -> Bool {
    a.layout == b.layout && a.layout == c.layout &&
        a.isBufferIterable && b.isBufferIterable && c.isBufferIterable
}

//==============================================================================
/// BufferElements
/// Iterates the buffer elements in order, independent of logical orientation
/// this is used for elementwise operations

// Note: this copies the host buffer so that it can be accessed asynchronously
// without copy-on-write issues
public struct BufferElements<Shape, TensorElement>: MutableCollection
    where Shape: TensorShape, TensorElement: StorageElement
{
    public let hostBuffer: UnsafeMutableBufferPointer<TensorElement.Stored>
    public let isSingleElement: Bool
    public let startIndex: Int
    public let endIndex: Int
    
    @inlinable public init(_ tensor: Tensor<Shape, TensorElement>) {
        assert(tensor.isBufferIterable)
        self.isSingleElement = tensor.storageSpanCount == 1
        let buff = tensor.storage.read(at: tensor.storageBase,
                                       count: tensor.storageSpanCount)
        // this does not actually mutate
        let p = UnsafeMutablePointer(mutating: buff.baseAddress)
        hostBuffer = UnsafeMutableBufferPointer(start: p, count: buff.count)
        startIndex = tensor.storageBase
        endIndex = startIndex + tensor.count
    }
    
    @inlinable public init(mutating tensor: Tensor<Shape, TensorElement>) {
        assert(tensor.isBufferIterable)
        self.isSingleElement = tensor.storageSpanCount == 1
        hostBuffer = tensor.storage.readWrite(at: tensor.storageBase,
                                              count: tensor.storageSpanCount)
        startIndex = tensor.storageBase
        endIndex = startIndex + tensor.count
    }
    
    @inlinable public func index(after i: Int) -> Int {
        i + 1
    }
    
    @inlinable public subscript(position: Int) -> TensorElement.Value {
        get {
            if isSingleElement {
                return TensorElement.value(from: hostBuffer[0], at: 0)
            } else {
                let si = TensorElement.storedIndex(position)
                return TensorElement.value(from: hostBuffer[si], at: position)
            }
        }
        
        set(newValue) {
            if isSingleElement {
                TensorElement.store(value: newValue, at: 0, to: &hostBuffer[0])
            } else {
                let si = TensorElement.storedIndex(position)
                TensorElement.store(value: newValue, at: position,
                                    to: &hostBuffer[si])
            }
        }
    }
}

//==============================================================================
/// StridedElements
/// Iterates storage elements using logical index coordinates and strides
public struct StridedElements<Shape, TensorElement>: MutableCollection
where Shape: TensorShape, TensorElement: StorageElement
{
    public typealias Index = ElementIndex<Shape>
    public typealias Element = TensorElement.Value
    public let hostBuffer: UnsafeMutableBufferPointer<TensorElement.Stored>
    public let spanCount: Int
    public let logicalStrides: Shape
    public let strides: Shape
    public let startIndex: Index
    public let endIndex: Index
    
    @inlinable public init(_ tensor: Tensor<Shape, TensorElement>) {
        self.logicalStrides = tensor.shape.strides(for: tensor.layout)
        self.strides = tensor.strides
        self.spanCount = tensor.shape.spanCount(stridedBy: strides)
        let buff = tensor.storage.read(at: tensor.storageBase, count: spanCount)
        // this does not actually mutate
        let p = UnsafeMutablePointer(mutating: buff.baseAddress)
        hostBuffer = UnsafeMutableBufferPointer(start: p, count: buff.count)
        startIndex = Index(Shape.zero, tensor.storageBase)
        endIndex = Index(tensor.shape, tensor.storageBase + tensor.count)
    }
    
    @inlinable public init(mutating tensor: Tensor<Shape, TensorElement>) {
        self.logicalStrides = tensor.shape.strides(for: tensor.layout)
        self.strides = tensor.strides
        self.spanCount = tensor.shape.spanCount(stridedBy: strides)
        hostBuffer = tensor.storage.readWrite(at: tensor.storageBase,
                                              count: spanCount)
        startIndex = Index(Shape.zero, tensor.storageBase)
        endIndex = Index(tensor.shape, tensor.storageBase + tensor.count)
    }
    
    //--------------------------------------------------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position,  position.index(stridedBy: logicalStrides))
    }

    @inlinable public func index(after i: Index) -> Index {
        i.incremented(between: startIndex, and: endIndex)
    }
    
    @inlinable public subscript(position: Index) -> Element {
        get {
            let i = position.linearIndex(strides)
            let si = TensorElement.storedIndex(i)
            return TensorElement.value(from: hostBuffer[si], at: i)
        }
        
        set(newValue) {
            let i = position.linearIndex(strides)
            let si = TensorElement.storedIndex(i)
            TensorElement.store(value: newValue, at: i, to: &hostBuffer[si])
        }
    }
}
