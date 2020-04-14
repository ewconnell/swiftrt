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

    // properties
    /// the diagnostic name for the collection
    @inlinable public static var name: String { "Tensor\(Shape.rank)" }
    /// the element storage buffer.
    @usableFromInline var storage: StorageBufferType<Element>
    /// the dense number of elements in the shape
    public let elementCount: Int
    /// the storage buffer base offset where this tensor's elements begin
    public let baseOffset: Int
    /// `true` if elements are in row major contiguous order
    // this is a stored property, because it's used during
    // gpu dispatch decision making
    public let isSequential: Bool
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder
    /// the dimensions of the element space
    public let shape: Shape
    // used by makeIndex
    @usableFromInline var shapeStrides: Shape { shape.sequentialStrides() }
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape

    //-----------------------------------
    /// `true` if the view will be shared by by multiple writers
    @inlinable public var isShared: Bool { _isShared }
    @usableFromInline var _isShared: Bool

    //-----------------------------------
    /// the starting index zero relative to the storage buffer
    public let startIndex: Index
    /// the ending index zero relative to the storage buffer
    public let endIndex: Index

    /// `true` if the tensor represents a single constant Element
    @inlinable public var isSingleElement: Bool { spanCount == 1 }

    //--------------------------------------------------------------------------
    /// init(
    /// Used to initialize a collection of dense stored elements
    @inlinable public init(
        shape: Shape,
        strides: Shape,
        elementCount: Int,
        spanCount: Int,
        storage: StorageBufferType<Element>,
        baseOffset: Int,
        order: StorageOrder,
        share: Bool,
        isSequential: Bool
    ) {
        self.shape = shape
        self.strides = strides
        self.elementCount = elementCount
        self.spanCount = spanCount
        self.storage = storage
        self.baseOffset = baseOffset
        self.storageOrder = order
        self._isShared = share
        self.isSequential = isSequential
        self.startIndex = Index(Shape.zero, baseOffset)
        self.endIndex = Index(shape, baseOffset + elementCount)
    }
    
    //--------------------------------------------------------------------------
    /// init(element:shape:
    /// Used to initialize a tensor with a single Element
    @inlinable public init(single element: Element, shape: Shape) {
        self.shape = shape
        strides = Shape.zero
        elementCount = shape.elementCount()
        spanCount = 1
        baseOffset = 0
        storageOrder = .C
        _isShared = true
        isSequential = true
        startIndex = Index(Shape.zero, 0)
        endIndex = Index(shape, elementCount)
        storage = StorageBufferType<Element>(single: element, name: "")
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
    @inlinable var flat: [Element] {
        [Element](self)
    }
    
    //--------------------------------------------------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position, position.index(stridedBy: shapeStrides))
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
    @inlinable subscript(i: Index) -> Element {
        get {
            // a single element can skip doing the buffer linear address
            // calculation. This is beneficial ranked higher ranked
            // repeated scalars. In perf, the cost of the branch appears
            // to be undetectable
            if isSingleElement {
                return storage.element(at: baseOffset)
            } else if isSequential {
                // most tensors are layed out sequentially, so it is much
                // cheaper to use the sequencePosition
                return storage.element(at: i.sequencePosition)
            } else {
                // perform a full strided buffer index calculation
                return storage.element(at: baseOffset + i.linearIndex(strides))
            }
        }
        set {
            if isSingleElement {
                return storage.setElement(value: newValue, at: baseOffset)
            } else if isSequential {
                storage.setElement(value: newValue, at: i.sequencePosition)
            } else {
                storage.setElement(value: newValue,
                                   at: baseOffset + i.linearIndex(strides))
            }
        }
    }

    //--------------------------------------------------------------------------
    // view subscript
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get { createView(lower, upper) }
        set {
            var view = createView(lower, upper)
            copy(from: newValue, to: &view)
        }
    }

    @inlinable func createView(_ lower: Shape, _ upper: Shape) -> Self {
        let shape = upper &- lower
        let isSeq = strides.areSequential(for: shape)
        let count = shape.elementCount()
        let span = isSeq ? count : shape.spanCount(stridedBy: strides)
        return Tensor(
            shape: shape,
            strides: strides,
            elementCount: count,
            spanCount: span,
            storage: storage,
            baseOffset: baseOffset + lower.index(stridedBy: strides),
            order: storageOrder,
            share: isShared,
            isSequential: isSeq)
    }
    
//    //--------------------------------------------------------------------------
//    // stepped view subscript
//    @inlinable subscript(lower: Shape, upper: Shape, steps: Shape) -> Self {
//        get {
//            (steps == Shape.one) ?
//                createView(lower, upper) :
//                createView(lower, upper, steps)
//        }
//        set {
//            var view = (steps == Shape.one) ?
//                createView(lower, upper) :
//                createView(lower, upper, steps)
//            copy(from: newValue, to: &view)
//        }
//    }
//
//    @inlinable
//    func createView(_ lower: Shape, _ upper: Shape, _ steps: Shape) -> Self {
//        let isSeq = false
//        var absSteps = steps
//        for i in 0..<Shape.rank { absSteps[i] = Swift.abs(absSteps[i]) }
//        var shape = upper &- lower
//        shape = ((shape &- 1 &+ absSteps) / absSteps) &+ lower
//        let strides = self.strides &* steps
//        let count = shape.elementCount()
//        let span = isSeq ? count : shape.spanCount(stridedBy: strides)
//        assert(shape.min() > 0, _messageInvalidShape)
//
//        return Tensor(
//            shape: shape,
//            strides: strides,
//            elementCount: count,
//            spanCount: span,
//            storage: storage,
//            baseOffset: baseOffset + lower.index(stridedBy: strides),
//            order: storageOrder,
//            share: isShared,
//            isSequential: isSeq)
//    }

    //--------------------------------------------------------------------------
    /// shared(
    @inlinable mutating func shared() -> Self {
        // if not uniquely held then copy before creating the shared view
        if !isKnownUniquelyReferenced(&storage) {
            diagnostic("\(mutationString) \(storage.name)(\(storage.id)) " +
                "\(Element.self)[\(elementCount)]",
                categories: [.dataCopy, .dataMutation])

            storage = StorageBufferType(copying: storage)
        }
        
        // copy self and set the isShared flag to true
        var sharedDense = self
        sharedDense._isShared = true
        return sharedDense
    }
}

//==============================================================================
// Tensor read write extensions
public extension Tensor {
    @inlinable func read() {
        
    }
    
    @inlinable func read(using queue: DeviceQueue) {
    }

    @inlinable mutating func readWrite() {
    }
    
    @inlinable mutating func readWrite(using queue: DeviceQueue) {
    }
}

//==============================================================================
// Tensor element properties
public extension Tensor {
    /// first
    /// - Returns: the first element in the tensor
    @_semantics("autodiff.nonvarying")
    @inlinable var first: Element {
        storage.element(at: baseOffset)
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @_semantics("autodiff.nonvarying")
    @inlinable var element: Element {
        get {
            assert(elementCount == 1, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return storage.element(at: baseOffset)
        }
        set {
            assert(elementCount == 1, "the `element` property expects " +
                "the tensor to have a single Element")
            storage.setElement(value: newValue, at: baseOffset)
        }
    }
}

