//******************************************************************************
// Copyright 2019 Google LLC
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
/// Tensor protocol
/// an n-dimensional collection of elements
public protocol Tensor: Collection, CustomStringConvertible, Logging
{
    /// the ranked short vector type that defines the collection's dimensions
    associatedtype Shape: TensorShape
    /// the type of element in the collection
    associatedtype Element

    //----------------------------------
    /// the number of elements described by `shape`
    var elementCount: Int { get }
    /// a label for the type used as a default name in diagnostics
    static var name: String { get }
    /// the dimensions of the collection
    var shape: Shape { get }
    /// the order in memory to store materialized Elements. Generator
    /// tensor types maintain this property as a template for dense
    /// result tensors.
    var storageOrder: StorageOrder { get }
    
    //----------------------------------
    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: the collection slice
    subscript(lower: Shape, upper: Shape) -> Self { get }
}

//==============================================================================
/// MutableTensor
/// an n-dimensional mutable collection of stored elements
public protocol MutableTensor: Tensor, MutableCollection
{
    /// `true` if the collection can be shared by multiple writers
    /// without performing copy-on-write
    var isShared: Bool { get }
    
    //----------------------------------
    /// shared
    /// returns a copy of `self` that does not perform copy-on-write to enable
    /// multi-threaded writes. If the associated storage is not uniquely
    /// referenced, then a copy will be made before returning the sharable
    /// copy. Subscripted views inherit the `isShared` property
    /// - Returns: a sharable copy of `self`
    mutating func shared() -> Self

    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: the collection slice
    subscript(lower: Shape, upper: Shape) -> Self { get set }
}

//==============================================================================
// default types
/// the type used for memory indexing on discreet devices
public typealias DeviceIndex = Int32

//==============================================================================
/// StorageOrder
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
/// These names are following the numpy naming convention
public enum StorageOrder: Int, Codable {
    /// C style row major memory layout
    case C
    /// Fortran style column major memory layout
    case F
    /// more expressive aliases
    public static let rowMajor = C, colMajor = F
}

//==============================================================================
/// ElementIndex
/// Common index type used to iterate through collection elements
/// `position` is the index position in n-dimensional space
/// `sequencePosition` is the linear sequence position when iterating
/// and used for comparison
public struct ElementIndex<Shape>: Comparable, Codable
    where Shape: TensorShape
{
    /// the logical position along each axis
    public let position: Shape
    /// linear sequence position
    public let sequencePosition: Int

    // init(position:sequencePosition:
    @inlinable public init(_ position: Shape, _ sequencePosition: Int) {
        self.position = position
        self.sequencePosition = sequencePosition
    }
    
    /// incremented(lower:upper:
    /// increments `position` with the range `lower..<upper`
    @inlinable
    public func incremented(between lower: Self, and upper: Self) -> Self {
        let pos = position.incremented(between: lower.position,
                                       and: upper.position)
        return ElementIndex(pos, sequencePosition + 1)
    }
    
    @inlinable public func linearIndex(with strides: Shape) -> Int {
        position.linearIndex(with: strides)
    }

    // Equatable
    @inlinable public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition == rhs.sequencePosition
    }
    
    // Comparable
    @inlinable public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition < rhs.sequencePosition
    }
}

//==============================================================================
/// Tensor extensions
public extension Tensor {
    @inlinable var flatArray: [Element] {
        [Element](self)
    }
}
