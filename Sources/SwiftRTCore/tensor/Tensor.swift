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
/// Tensor
public struct Tensor<Shape, Element>: MutableTensorType
    where Shape: TensorShape
{
    // types
    public typealias Index = ElementIndex<Shape>
    /// the storage buffer base offset where this tensor's elements begin
    public let baseOffset: Int
    /// the dense number of elements in the shape
    public let count: Int
    /// the unique storage id
    @inlinable public var id: Int { storage.id }
    /// `true` if elements are in row major contiguous order
    // this is a stored property, because it's used during
    // gpu dispatch decision making
    public let isSequential: Bool
    /// the dimensions of the element space
    @noDerivative public let shape: Shape
    // used by makeIndex
    public let shapeStrides: Shape
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape
    /// the element storage buffer.
    public var storage: StorageBufferType<Element>
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder

    //-----------------------------------
    /// `true` if the view will be shared by by multiple writers
    @inlinable public var isShared: Bool { _isShared }
    public var _isShared: Bool

    //-----------------------------------
    /// the starting index zero relative to the storage buffer
    public let startIndex: Index
    /// the ending index zero relative to the storage buffer
    public let endIndex: Index

    /// `true` if the tensor represents a single constant Element
    @inlinable public var isSingleElement: Bool { spanCount == 1 }
    
    /// `true` if the tensor elements are densely packed
    @inlinable public var isContiguous: Bool { count == spanCount }

    /// the name of the collection
    @inlinable public var name: String {
        get { storage.name }
        set { storage.name = newValue }
    }

    //--------------------------------------------------------------------------
    /// init(
    /// Used to initialize a collection of dense stored elements
    @inlinable public init(
        shape: Shape,
        strides: Shape,
        count: Int,
        spanCount: Int,
        storage: StorageBufferType<Element>,
        baseOffset: Int,
        order: StorageOrder,
        share: Bool,
        isSequential: Bool
    ) {
        self.shape = shape
        self.strides = strides
        self.count = count
        self.spanCount = spanCount
        self.storage = storage
        self.baseOffset = baseOffset
        self.storageOrder = order
        self._isShared = share
        self.isSequential = isSequential
        self.startIndex = Index(Shape.zero, baseOffset)
        self.endIndex = Index(shape, baseOffset + count)
        self.shapeStrides = shape.strides(for: order)
    }
    
    //--------------------------------------------------------------------------
    /// init(element:shape:
    /// Used to initialize a tensor with a single Element
    @inlinable public init(single element: Element, shape: Shape) {
        self.shape = shape
        strides = Shape.zero
        count = shape.elementCount()
        spanCount = 1
        baseOffset = 0
        storageOrder = .C
        _isShared = true
        isSequential = true
        startIndex = Index(Shape.zero, 0)
        endIndex = Index(shape, count)
        storage = StorageBufferType<Element>(single: element)
        storage.name = "Element"
        shapeStrides = Shape.zero
    }
}

//==============================================================================
/// DifferentiableTensor
///
/// While these protocols are not strictly necessary, they are used
/// to reduce the number of generic requirements when writing
/// `@differentiable` attributes
///
public protocol DifferentiableTensor: TensorType & Differentiable
    where Self == TangentVector, Element: DifferentiableElement {}

/// DifferentiableElement
public protocol DifferentiableElement:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

// this is defined with the typealias because of AD same file
// compiler requirements. Hopefully fixed in the future
extension Complex: DifferentiableElement {
  public typealias TangentVector = Self
}

// Differentiable conformance
extension Tensor: Differentiable & DifferentiableTensor
    where Element: DifferentiableElement
{
    public typealias TangentVector = Self
}

extension Tensor: AdditiveArithmetic where Element: Numeric {
    @inlinable public static var zero: Self { Tensor(0) }
    @inlinable public static var one: Self { Tensor(1) }
}

//==============================================================================
// Tensor Codable
public enum TensorCodingKeys: String, CodingKey { case data, shape, name }

extension Tensor: Codable where Element: Codable {
    /// encodes the contents of the array
    @inlinable public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TensorCodingKeys.self)
        try container.encode(storage.name, forKey: .name)
        try container.encode(shape, forKey: .shape)
        var dataContainer = container.nestedUnkeyedContainer(forKey: .data)
        try self.forEach {
            try dataContainer.encode($0)
        }
    }
    
    @inlinable public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TensorCodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        let shape = try container.decode(Shape.self, forKey: .shape)
        var dataContainer = try container.nestedUnkeyedContainer(forKey: .data)
        self = Self(shape)
        self.name = name

        assert(self.count == dataContainer.count)
        for i in self.indices {
            self[i] = try dataContainer.decode(Element.self)
        }
    }
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
// Tensor collection and sub view extensions
public extension Tensor {
    //--------------------------------------------------------------------------
    /// - Returns: the collection elements as a 1D Swift array
    @inlinable var flatArray: [Element] {
        [Element](self)
    }
    
    //--------------------------------------------------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position, baseOffset + position.index(stridedBy: shapeStrides))
    }

    //--------------------------------------------------------------------------
    /// index(i:
    @inlinable func index(after i: Index) -> Index {
        // If the storage is a single broadcast element or many, the index
        // still has to be incremented to satisfy reaching the end of
        // the collection
        if isSequential {
            return Index(at: i.sequencePosition &+ 1)
        } else {
            return i.incremented(between: startIndex, and: endIndex)
        }
    }

    //--------------------------------------------------------------------------
    // elemment subscript
    // NOTE: with only a single tensor type allowed, lifting the branch out
    // of this per index function turned out to have worse performance
    // because it prevented inlining cross module. As is, the branches
    // don't appear to affect perf numbers.
    @inlinable subscript(i: Index) -> Element {
        get {
            // a single element can skip doing the buffer linear address
            // calculation. This is beneficial ranked higher ranked
            // repeated scalars.
            if isSingleElement {
                return storage.element(at: baseOffset)
            } else if isSequential {
                // most tensors are layed out sequentially, so it is much
                // cheaper to use the sequencePosition. The `baseOffset`
                // is included during index initialization
                return storage.element(at: i.sequencePosition)
            } else {
                // perform a full strided buffer index calculation
                return storage.element(at: baseOffset &+ i.linearIndex(strides))
            }
        }
        
        set {
            if isSingleElement {
                return storage.setElement(value: newValue, at: baseOffset)
            } else if isSequential {
                storage.setElement(value: newValue, at: i.sequencePosition)
            } else {
                storage.setElement(value: newValue,
                                   at: baseOffset &+ i.linearIndex(strides))
            }
        }
    }

    //--------------------------------------------------------------------------
    // sub view subscript
    @differentiable(where Element: DifferentiableElement)
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get { createView(lower, upper, isShared) }
        set {
            ensureValidStorage()
            var view = createView(lower, upper, true)
            copy(from: newValue, to: &view)
        }
    }

    @inlinable
    func createView(_ lower: Shape, _ upper: Shape, _ share: Bool) -> Self {
        let shape = upper &- lower
        let isSeq = strides.areSequential(for: shape)
        let count = shape.elementCount()
        let span = isSeq ? count : shape.spanCount(stridedBy: strides)
        return Tensor(
            shape: shape,
            strides: strides,
            count: count,
            spanCount: span,
            storage: storage,
            baseOffset: baseOffset + lower.index(stridedBy: strides),
            order: storageOrder,
            share: share,
            isSequential: isSeq)
    }

    //--------------------------------------------------------------------------
    /// shared(
    @inlinable mutating func shared() -> Self {
        // copy self and set the isShared flag to true
        var sharedDense = self
        sharedDense._isShared = true
        return sharedDense
    }

    //--------------------------------------------------------------------------
    /// `ensureValidStorage`
    /// called before a write operation to ensure that the storage buffer
    /// is unique for this tensor unless it `isShared`
    /// It also expands repeated tensors to a full dense storage
    /// representation for write, which most often happens via element
    /// subscripting.
    @inlinable mutating func ensureValidStorage() {
        // if repeated then expand to full dense tensor
        if spanCount < count {
            var expanded = Tensor(like: self)

            diagnostic(
                "\(expandingString) \(name)(\(id)) " +
                    "\(Element.self)[\(spanCount)] to: \(expanded.name)" +
                    "(\(expanded.id)) \(Element.self)[\(expanded.count)]",
                categories: [.dataCopy, .dataExpanding])

            // do an indexed copy
            copy(from: self, to: &expanded)
            self = expanded

        } else if !(isKnownUniquelyReferenced(&storage) || isShared) {
            // if not uniquely held then copy before creating the shared view
            diagnostic("\(mutationString) \(storage.name)(\(storage.id)) " +
                        "\(Element.self)[\(count)]",
                       categories: [.dataCopy, .dataMutation])
            
            storage = StorageBufferType(copying: storage)
        }
    }
}

//==============================================================================
/// Derivative registration
extension Tensor where Element: DifferentiableElement {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    
    @derivative(of: subscript)
    @inlinable func _vjpSubscript(lower: Shape, upper: Shape)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper], { v in
            var result = zeros(like: self)
            result[lower, upper] = v
            return result
        })
    }
}

//==============================================================================
// Tensor read write access
public extension Tensor {
    //--------------------------------------------------------------------------
    /// `read`
    /// Synchronizes the collection of materialized elements for reading.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `Collection`
    /// enumeration via `indices` or integer subscripting.
    @inlinable func read() {
        
    }
    
    //--------------------------------------------------------------------------
    /// `read(queue:
    /// Synchronizes a collection of materialized elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable func read(using queue: DeviceQueue) {
    }

    //--------------------------------------------------------------------------
    /// `readWrite`
    /// Synchronizes a collection of materialized elements for read write.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `MutableCollection`
    /// enumeration via `indices` or subscripting.
    @inlinable mutating func readWrite() {
        ensureValidStorage()
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite(queue:`
    /// Synchronizes a mutable collection of materialized elements
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable mutating func readWrite(using queue: DeviceQueue) {
        ensureValidStorage()
    }
}

//==============================================================================
// Tensor element properties
public extension Tensor {
    /// first
    /// - Returns: the first element in the tensor
    @inlinable var first: Element {
        storage.element(at: baseOffset)
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @differentiable(where Element: DifferentiableElement)
    @inlinable var element: Element {
        get {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return storage.element(at: baseOffset)
        }
        set {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element")
            storage.setElement(value: newValue, at: baseOffset)
        }
    }

    @derivative(of: element)
    @inlinable func vjpElement() -> (
      value: Element,
      pullback: (Element) -> Self
    ) where Element: DifferentiableElement {
      (element, { v in
        var result = zeros(like: self)
        result.element = v
        return result
      })
    }
}

