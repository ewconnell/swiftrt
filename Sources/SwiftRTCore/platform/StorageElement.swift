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
    public typealias Stored = Self
    public typealias Value = Float
    
    @inlinable public static func value(
        from stored: Stored, at index: Int
    ) -> Value {
        Value(stored)
    }
    
    @inlinable public static func store(
        value: Value, at index: Int, to stored: inout Stored
    ) {
        stored = Stored(value)
    }
    
    @inlinable public static func stored(value: Value) -> Stored {
        Stored(value)
    }
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

extension Int: StorageElement {
    public typealias Stored = Self
    public typealias Value = Self
}

extension UInt: StorageElement {
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

//==============================================================================
/// BufferSequential
/// walks the buffer elements in order, independent of logical orientation
/// this is used for elementwise operations

// Note: we copy the host buffer so that it can be accessed asynchronously
// without copy-on-write issues
public struct BufferSequential<Shape, TensorElement>: MutableCollection
    where Shape: TensorShape, TensorElement: StorageElement
{
    public typealias Index = ElementIndex<Shape>
    public typealias Element = TensorElement.Value
    public let hostBuffer: UnsafeMutableBufferPointer<TensorElement.Stored>
    public let isSingleElement: Bool
    public let startIndex: Index
    public let endIndex: Index
    
    @inlinable public init(_ tensor: Tensor<Shape, TensorElement>) {
        hostBuffer = tensor.storage.hostBuffer
        startIndex = tensor.startIndex
        endIndex = tensor.endIndex
        isSingleElement = tensor.count == 1
    }
    
    @inlinable public init(mutating tensor: Tensor<Shape, TensorElement>) {
        self.init(tensor)
    }
    
    @inlinable public init(
        mutating tensor: Tensor<Shape, TensorElement>,
        _ index: Index, _ newValue: Element
    ) {
        self.init(tensor)
        self[index] = newValue
    }
    
    @inlinable public func index(after i: Index) -> Index {
        Index(at: i.sequencePosition &+ 1)
    }
    
    @inlinable public subscript(position: Index) -> Element {
        get {
            if isSingleElement {
                return TensorElement.value(from: hostBuffer[0], at: 0)
            } else {
                let i = position.sequencePosition
                let si = TensorElement.storedIndex(i)
                return TensorElement.value(from: hostBuffer[si], at: i)
            }
        }
        
        set(newValue) {
            if isSingleElement {
                TensorElement.store(value: newValue, at: 0, to: &hostBuffer[0])
            } else {
                let i = position.sequencePosition
                let si = TensorElement.storedIndex(i)
                TensorElement.store(value: newValue, at: i, to: &hostBuffer[si])
            }
        }
    }
}

//==============================================================================
/// RowSequential
/// walks the buffer elements in logical row major order
public typealias RowSequential<Shape, TensorElement> =
    BufferSequential<Shape, TensorElement>
where Shape: TensorShape, TensorElement: StorageElement

//==============================================================================
/// ColSequential
/// walks the col major buffer elements in logical row major order
public struct ColSequential<Shape, TensorElement>: MutableCollection
where Shape: TensorShape, TensorElement: StorageElement
{
    public typealias Index = ElementIndex<Shape>
    public typealias Element = TensorElement.Value
    public let hostBuffer: UnsafeMutableBufferPointer<TensorElement.Stored>
    public let isSingleElement: Bool
    public let startIndex: Index
    public let endIndex: Index
    
    @inlinable public init(_ tensor: Tensor<Shape, TensorElement>) {
        hostBuffer = tensor.storage.hostBuffer
        startIndex = tensor.startIndex
        endIndex = tensor.endIndex
        isSingleElement = tensor.count == 1
    }
    
    @inlinable public init(mutating tensor: Tensor<Shape, TensorElement>) {
        self.init(tensor)
    }
    
    @inlinable public init(
        mutating tensor: Tensor<Shape, TensorElement>,
        _ index: Index, _ newValue: Element
    ) {
        self.init(tensor)
        self[index] = newValue
    }
    
    @inlinable public func index(after i: Index) -> Index {
        Index(at: i.sequencePosition &+ 1)
    }
    
    @inlinable public subscript(position: Index) -> Element {
        get {
            let i = position.sequencePosition
            let si = TensorElement.storedIndex(i)
            return TensorElement.value(from: hostBuffer[si], at: i)
        }
        set(newValue) {
            let i = position.sequencePosition
            let si = TensorElement.storedIndex(i)
            TensorElement.store(value: newValue, at: i, to: &hostBuffer[si])
        }
    }
}

