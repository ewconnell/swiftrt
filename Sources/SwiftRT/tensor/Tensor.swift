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
import Numerics

//==============================================================================
/// Tensor protocol
/// an n-dimensional collection of elements
public protocol Tensor: Collection, CustomStringConvertible, Logging
    where Index == ElementIndex<Shape>
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
    // for guaranteed discreet device compatibility
    /// - Returns: a value if the tensor can be represented as a
    /// single element, and `nil` if it cannot.
    var asElement: Element? { get }
    
    /// used to ensure that a discreet device always has something
    /// it can work with. In the case of an unrecognized generator tensor,
    /// this can be used to render it on the cpu into a common form.
    var asDense: DenseTensor<Shape, Element> { get }

    //----------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    func makeIndex(at position: Shape) -> Index

    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: the collection slice
    subscript(lower: Shape, upper: Shape) -> Self { get }

    //----------------------------------
    /// `read`
    /// Synchronizes a collection of materialized elements for reading.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `Collection`
    /// enumeration via `indices` or subscripting.
    func read()
    
    /// `read(queue:
    /// Synchronizes a collection of materialized elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    func read(using queue: DeviceQueue)
}

//------------------------------------------------------------------------------

public extension Tensor {
    // data is always available for generator tensor types
    // so provide no op stubs for them
    @inlinable func read() { }
    @inlinable func read(using queue: DeviceQueue) { }
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
    /// `shared`
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
    
    //----------------------------------
    /// `readWrite`
    /// Synchronizes a collection of materialized elements for read write.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `MutableCollection`
    /// enumeration via `indices` or subscripting.
    mutating func readWrite()

    /// `readWrite(queue:`
    /// Synchronizes a mutable collection of materialized elements
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    mutating func readWrite(using queue: DeviceQueue)
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

    /// init(sequencePosition:
    /// initializer for collections that ignore logical position
    @inlinable public init(at sequencePosition: Int) {
        self.position = Shape.zero
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
    
    @inlinable public func linearIndex(_ strides: Shape) -> Int {
        position.index(stridedBy: strides)
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
// Tensor extensions
public extension Tensor {
    //--------------------------------------------------------------------------
    /// - Returns: the collection elements as a 1D Swift array
    @inlinable var flatArray: [Element] {
        [Element](self)
    }
}

//==============================================================================
/// DifferentiableTensor
///
/// Marker protocol for `Tensor` that conform to `Differentiable`.
///
/// While this protoocl is not strictly necessary, it is used to reduce the
/// number of generic requirements when writing `@differentiable` attributes on
/// generic differentiable `Tensor` functions.
public protocol DifferentiableTensor: Tensor & Differentiable
    where Self == TangentVector, Element: DifferentiableElement {}

//==============================================================================
/// DifferentiableElement
// this is for shorthand also to make the code less verbose
public protocol DifferentiableElement:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

// this is defined with the typealias because of AD same file
// compiler requirements. Hopefully fixed in the future
extension Complex: DifferentiableElement {
  public typealias TangentVector = Self
}
