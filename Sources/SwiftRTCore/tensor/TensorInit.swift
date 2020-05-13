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
// ranked convenience types
/// Tensor
public typealias TensorR1<Element> = Tensor<Shape1, Element>
public typealias TensorR2<Element> = Tensor<Shape2, Element>
public typealias TensorR3<Element> = Tensor<Shape3, Element>
public typealias TensorR4<Element> = Tensor<Shape4, Element>
public typealias TensorR5<Element> = Tensor<Shape5, Element>
public typealias TensorR6<Element> = Tensor<Shape6, Element>

//==============================================================================
// parameter matching helper
// TODO: THIS NEEDS TO BE REMOVED. IT'S A HACK FOR AD SUPPORT
@inlinable public func match<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (Tensor<S,E>, Tensor<S,E>) where S: TensorShape
{
    if lhs.count == rhs.count {
        return (lhs, rhs)
    } else if lhs.count > rhs.count {
        return (lhs, Tensor<S,E>(repeating: rhs, to: lhs.shape))
    } else {
        return (Tensor<S,E>(repeating: lhs, to: rhs.shape), rhs)
    }
}

//==============================================================================
// Tensor initializers
//==============================================================================

public extension Tensor {
    @inlinable var diagnosticName: String {
        "\(name)R\(Shape.rank)_(\(storage.id))"
    }

    //--------------------------------------------------------------------------
    /// init(shape:order:
    /// creates a dense shape
    /// - Parameters:
    ///  - shape: the n-dimensional shape of the tensor
    ///  - order: the storage order of the elements
    @inlinable init(_ shape: Shape, order: StorageOrder = .C) {
        let count = shape.elementCount()
        self.init(shape: shape,
                  strides: shape.strides(for: order),
                  count: count,
                  spanCount: count,
                  storage: StorageBufferType(count: count),
                  baseOffset: 0,
                  order: order,
                  share: false,
                  isSequential: order == .C)
    }
    
    //--------------------------------------------------------------------------
    /// init
    @inlinable init() {
        self.init(Shape.zero)
    }
    
    //--------------------------------------------------------------------------
    /// init(like:order:
    /// convenience initializer to initialize with the shape and type as `other`
    /// - Parameters:
    ///  - other: a tensor to copy attributes from
    ///  - order: the storage order of the elements
    @inlinable init(like other: Self, order: StorageOrder = .C) {
        self.init(other.shape, order: order)
    }
    
    //--------------------------------------------------------------------------
    /// init(element:
    /// creates a tensor with a single scalar value
    /// - Parameter element: the single element value for the tensor
    @inlinable init(_ element: Element) {
        self.init(single: element, shape: Shape.one)
    }

    //--------------------------------------------------------------------------
    /// init(repeating element:shape:
    /// Repeats a single stored element while indexing
    /// - Parameters:
    ///  - element: the element value to repeat while indexing
    ///  - shape: the shape of the tensor
    @differentiable(where Element: DifferentiableElement)
    @inlinable init(repeating element: Element, to shape: Shape) {
        self.init(single: element, shape: shape)
    }

    //--------------------------------------------------------------------------
    /// init(repeating other:shape:
    /// Repeats a tensor withing the specified shape while indexing
    /// - Parameters:
    ///  - other: the tensor to repeat
    ///  - shape: the shape of the tensor
    @differentiable(where Element: DifferentiableElement)
    @inlinable init(repeating other: Self, to shape: Shape) {
        // make sure the bounds are compatible
        assert({
            for i in 0..<Shape.rank {
                if other.shape[i] != 1 && shape[i] != other.shape[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor dimensions must be either 1" +
            " or match the repeated shape")

        // compute strides, setting stride to 0 for repeated dimensions
        var repeatedStrides = Shape.zero
        for i in 0..<Shape.rank where other.shape[i] == shape[i] {
            repeatedStrides[i] = other.strides[i]
        }
        let count = shape.elementCount()
        self.init(shape: shape,
                  strides: repeatedStrides,
                  count: count,
                  spanCount: shape.spanCount(stridedBy: repeatedStrides),
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: count == 1)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element == Element
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let buffer = storage.readWrite(at: 0, count: count)
        _ = buffer.initialize(from: elements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryInteger, Element: Numeric
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { value -> Element in
            assert(Element(exactly: value) != nil,
                   "Value cast \(Element.self)(\(value)) failed")
            return Element(exactly: value)!
        }
        let buffer = storage.readWrite(at: 0, count: count)
        _ = buffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection,
        C.Element: BinaryFloatingPoint, Element: BinaryInteger
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        let buffer = storage.readWrite(at: 0, count: count)
        _ = buffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection,
        C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        let buffer = storage.readWrite(at: 0, count: count)
        _ = buffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// reductionBounds
    /// returns the upper bounds for a reduction result along the specified axes
    @inlinable func reductionShape(alongAxes axes: Set<Int>?) -> Shape {
        guard let axes = axes else { return Shape.one }
        assert(axes.isSubset(of: 0..<Shape.rank), "axis is out of bounds")
        var result = shape
        // set 1 for each dimension specified in the axes list
        axes.forEach { result[$0] = 1 }
        return result
    }
}

extension Tensor where Element: DifferentiableElement
{
    @derivative(of: init(repeating:to:))
    @inlinable static func _vjpInit(repeating value: Element, to shape: Shape)
        -> (value: Self, pullback: (Self) -> (Element))
    {
        (Self(repeating: value, to: shape), { $0.sum().element })
    }
    
    @derivative(of: init(repeating:to:))
    @inlinable static func _vjpInit(repeating other: Self, to shape: Shape)
        -> (value: Self, pullback: (Self) -> (Self))
    {
        // TODO: this is probably wrong. Test this
        (Self(repeating: other, to: shape), { $0 })
    }
}


//==============================================================================
// Rank transformations
public extension Tensor {
    /// concatenated tensors
    @inlinable init(concatenating tensors: Self..., axis: Int = 0) {
        self = Self(concatenating: tensors, axis: axis)
    }
    
    @inlinable init(concatenating tensors: [Self], axis: Int = 0) {
        self = SwiftRTCore.concatenate(tensors, axis: axis)
    }
}

//==============================================================================
/// init(reshaping:shape:order:
/// - Parameters:
///  - other: the tensor to reshape
///  - newShape: the shape of the new tensor
///  - order: the storage order of the new tensor
public extension Tensor {

    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(
        reshaping other: Tensor<S,Element>,
        to newShape: Shape,
        order newOrder: StorageOrder = .C
    ) where S: TensorShape
    {
        assert(other.isContiguous, "cannot reshape non contiguous data")
        assert(
            {
                var found = false
                for i in 0..<Shape.rank where newShape[i] == -1 {
                    if found { return false }
                    found = true
                }
                return true
            }(), "There can only be one instance of -1 in the shape")

        // resolve an implied dimension if it exists
        var shape = newShape
        for i in 0..<shape.count where newShape[i] == -1 {
            shape[i] = 1
            let specifiedCount = shape.elementCount()
            assert(other.count % specifiedCount == 0,
                   "incompatible dimensions")
            shape[i] = other.count / specifiedCount
        }
        assert(shape.elementCount() == other.count,
               "the new shape must have the same number of elements as other")
        
        // determine storage order
        let order: StorageOrder = newOrder == .F ||
            (newOrder == .A &&
            other.storageOrder == .F &&
            other.isSequential) ? .F : .C

        // reorder other's elements if needed
        var source = other
        if order != other.storageOrder {
            // change the source to have the new storage order and copy
            let strides = other.shape.strides(for: order)
            source = Tensor<S,Element>(
                shape: other.shape,
                strides: strides,
                count: other.count,
                spanCount: other.spanCount,
                storage: StorageBufferType<Element>(count: source.count),
                baseOffset: 0,
                order: other.storageOrder,
                share: other.isShared,
                isSequential: strides.areSequential(for: other.shape))
            
            // performs an indexed copy which reorders the elements
            Context.local.platform.diagnostic(
                "\(reorderString) copying \(other.diagnosticName) --> " +
                    "\(source.diagnosticName) \(Element.self)[\(source.count)]",
                categories: [.dataCopy, .dataReorder])
            
            copy(from: other, to: &source)
        }
        
        // init with new shape in the corrected order
        self.init(shape: shape,
                  strides: shape.strides(for: order),
                  count: source.count,
                  spanCount: source.spanCount,
                  storage: source.storage,
                  baseOffset: source.baseOffset,
                  order: order,
                  share: source.isShared,
                  isSequential: source.isSequential)
    }
}

extension Tensor where Element: DifferentiableElement
{
    @derivative(of: init(reshaping:to:order:))
    @inlinable static func _vjpInit<S>(
        reshaping other: Tensor<S,Element>,
        to newShape: Shape,
        order newOrder: StorageOrder
    ) -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape
    {
        let value = Self(reshaping: other, to: newShape, order: newOrder)
        return (value, {
            Tensor<S,Element>(reshaping: $0, to: other.shape,
                              order: other.storageOrder)
        })
    }
}

//==============================================================================
/// init(expanding other:axes:
/// - Parameters:
///  - other: the tensor to expand
///  - axes: the list of axes to expand

public extension Tensor {
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S, Axes>(
        expanding other: Tensor<S,Element>,
        axes: Axes
    ) where S: TensorShape, Axes: TensorShape
    {
        // set the expanded axes
        var shape = Shape.zero
        var strides = Shape.zero

        // default case indents by 1
        if axes.count == 1 && axes[0] == 0 {
            shape[0] = 1
            strides[0] = other.shape[0] * other.strides[0]
            for (i, j) in zip(1..<Shape.rank, 0..<S.rank) {
                shape[i] = other.shape[j]
                strides[i] = other.strides[j]
            }
        } else {
            assert(Shape.rank == S.rank + axes.count, "rank mismatch")
            // set 1 in expanded dimensions making sure axes are positive
            // and keeping in mind the axes could be in any order
            for i in 0..<axes.count {
                shape[axes[i] >= 0 ? axes[i] : axes[i] + S.rank] = 1
            }
            
            var axis = Shape.rank - 1
            var otherAxis = S.rank - 1
            while axis >= 0 {
                if shape[axis] == 1 {
                    if axis == Shape.rank - 1 {
                        // if the last dimension is expanded, then stride is 1
                        strides[axis] = 1
                    } else {
                        // if inserted, then compute stride
                        strides[axis] = shape[axis + 1] * strides[axis + 1]
                    }
                } else {
                    // simply copy stride
                    shape[axis] = other.shape[otherAxis]
                    strides[axis] = other.strides[otherAxis]
                    otherAxis -= 1
                }
                axis -= 1
            }
        }
        
        self.init(shape: shape,
                  strides: strides,
                  count: other.count,
                  spanCount: other.spanCount,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: other.isSequential)
    }
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(expanding other: Tensor<S,Element>, axes: Int...)
        where S: TensorShape
    {
        self.init(expanding: other, axes: Shape(axes))
    }
}

extension Tensor where Element: DifferentiableElement
{
    @derivative(of: init(expanding:axes:))
    @inlinable public static func _vjpInit<S, Axes>(
        expanding other: Tensor<S,Element>,
        axes: Axes
    ) -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape, Axes: TensorShape
    {
        let value = Self(expanding: other, axes: axes)
        return (value, { Tensor<S,Element>(squeezing: $0, axes: axes) })
    }
}

//==============================================================================
/// init(squeezing:axes
/// - Parameters:
///  - other: the collection to squeeze
///  - axes: a list of axes to squeeze
public extension Tensor {
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S,Axes>(
        squeezing other: Tensor<S,Element>,
        axes: Axes
    ) where S: TensorShape, Axes: TensorShape
    {
        assert(Shape.rank == S.rank - Axes.rank, "rank mismatch")
        var axis = 0
        var shape = Shape.zero
        var strides = Shape.zero
        var otherShape = other.shape

        // zero the axes to squeeze. These are done in two loops
        // because the axes list could be in any order
        for i in 0..<axes.count {
            otherShape[axes[i] >= 0 ? axes[i] : axes[i] + S.rank] = 0
        }
        
        for i in 0..<S.rank where otherShape[i] > 0 {
            shape[axis] = otherShape[i]
            strides[axis] = other.strides[i]
            axis += 1
        }
        
        self.init(shape: shape,
                  strides: strides,
                  count: other.count,
                  spanCount: other.count,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: other.isSequential)
    }
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(squeezing other: Tensor<S,Element>, axes: Int...)
        where S: TensorShape
    {
        self.init(squeezing: other, axes: Shape(axes))
    }
}

extension Tensor where Element: DifferentiableElement
{
    @derivative(of: init(squeezing:axes:))
    @inlinable static func _vjpInit<S, Axes>(
        squeezing other: Tensor<S,Element>,
        axes: Axes
    ) -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape, Axes: TensorShape
    {
        let value = Self(squeezing: other, axes: axes)
        return (value, { Tensor<S,Element>(expanding: $0, axes: axes) })
    }
}

//==============================================================================
/// init(stacking:axis:
/// - Parameters:
///  - others: the collection to squeeze
///  - axis: the axis to stack along
public extension Tensor {

    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(
        stacking others: [Tensor<S,Element>],
        axis: Int = 0
    ) where S: TensorShape
    {
        // make positive
        let positiveAxis = axis < 0 ? axis + S.rank : axis
        // create tensor of stacked shape and copy
        self = withoutDerivative(
            at: Self(stackedShape(of: others, along: positiveAxis)))
        stack(others, axis: positiveAxis, into: &self)
    }
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(stacking others: Tensor<S,Element>..., axis: Int = 0) {
        self.init(stacking: others, axis: axis)
    }
}

//==============================================================================
/// stackedShape(
// get the stacked shape with the inserted axis
@inlinable func stackedShape<S,SR,E>(
    of tensors: [Tensor<S,E>],
    along axis: Int = 0
) -> SR where S: TensorShape, SR: TensorShape
{
    assert(axis >= 0)
    var j = 0
    var stackedShape = SR.zero
    stackedShape[axis] = tensors.count
    for i in 0..<SR.rank where i != axis {
        stackedShape[i] = tensors[0].shape[j]
        j += 1
    }
    return stackedShape
}

//==============================================================================
/// stack(_:axis:into:
/// - Parameters:
///  - others: the collection to squeeze
///  - axis: the axis to stack along
///  - result: the output tensor
@differentiable(where E: DifferentiableElement)
@inlinable public func stack<S,SR,E>(
    _ tensors: [Tensor<S,E>],
    axis: Int = 0,
    into result: inout Tensor<SR,E>
) where S: TensorShape, SR: TensorShape
{
    // make positive
    let axis = axis < 0 ? axis + SR.rank : axis

    // verify that tensors are the correct rank and same shape
    assert(tensors.count > 0 && S.rank == SR.rank - 1,
           "stacked tensors must be one less than result rank \(S.rank - 1)")
    assert({
        let shape = tensors[0].shape
        for i in 1..<tensors.count {
            if tensors[i].shape != shape { return false }
        }
        return true
    }(), "stacked tensors must all be the same size")
    assert(result.shape == stackedShape(of: tensors, along: axis),
           "result tensor does not match the stacked shape of the inputs")
    
    // expand dimensions of each input along axis
    let expanded = tensors.map {
        Tensor<SR,E>(expanding: $0, axes: Shape1(axis))
    }
    
    // copy others into place
    var lower = SR.zero
    for tensor in expanded {
        result[lower, lower &+ tensor.shape] = tensor
        lower[axis] += 1
    }
}

@derivative(of: stack)
func vjpStack<S,SR,E>(
    _ tensors: [Tensor<S,E>],
    axis: Int = 0,
    into result: inout Tensor<SR,E>
) -> (value: (), pullback: (inout Tensor<SR, E>.TangentVector)
        -> Array<Tensor<S, E>>.TangentVector)
where S: TensorShape, SR: TensorShape
{
    let tensorCount = tensors.count
    func pullback(_ resultTangent: inout Tensor<SR, E>.TangentVector)
    -> Array<Tensor<S, E>>.TangentVector
    {
        // Fill `tensorTangents` with slices of `resultTangent` of shape
        // `tensorShapes[0]`, `tensorShapes[1]`, etc.
        var tensorTangents: [Tensor<S, E>] = []
        var lower = SR.zero
        var upper = resultTangent.shape
        upper[axis] = 1
        for _ in 0..<tensorCount {
            let slice = Tensor<S,E>(squeezing: resultTangent[lower, upper],
                                    axes: Shape1(axis))
            tensorTangents.append(slice)
            lower[axis] += 1
            upper[axis] += 1
        }

        // Set `resultTangent` to zero.
        // Note: We can't use `fill(_:with:)` because `resultTangent` aliases
        // `tensorTangents`.
        // TODO: track and fix
        // Note: https://bugs.swift.org/browse/TF-1250 will allow us to make
        // this pullback more efficient. How:
        // - Set the wrt parameters and results to
        //     @differentiable(wrt: (tensors), results: (result))
        // - This makes `resultTangent` not be inout, so we don't need to set
        //   it any more.
        resultTangent = zeros(like: resultTangent)

        return Array.DifferentiableView(tensorTangents)
    }
    return (stack(tensors, axis: axis, into: &result), pullback)
}

//==============================================================================
/// init(indenting:
///
public extension Tensor {

    @inlinable init<S>(indenting other: Tensor<S,Element>)
        where S: TensorShape
    {
        assert(S.rank < Shape.rank, "can only indent lower ranked shapes")

        // Self and other are different ranks so we append other's elements
        let start = Shape.rank - S.rank
        var shape = Shape.one
        var strides = Shape.one
        for (i, j) in zip(start..<Shape.rank, 0..<S.rank) {
            shape[i] = other.shape[j]
            strides[i] = other.strides[j]
        }
        for i in 0..<start {
            strides[i] = other.strides[0]
        }

        //-----------------------------------
        self.init(shape: shape,
                  strides: strides,
                  count: other.count,
                  spanCount: other.count,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: other.isSequential)
    }
}

//==============================================================================
/// init(transposing:permutations:
/// Returns a new data shape where the bounds and strides are permuted
/// - Parameter permutations: the indice order mapping. `count` must
///   equal `rank`
/// - Returns: transposed/permuted shape
/// - Precondition: Each value in `permutations` must be in the range
///   `-rank..<rank`
public extension Tensor {

    @differentiable(where Element: DifferentiableElement)
    @inlinable init(
        transposing other: Self,
        permutatedBy permutations: Shape? = nil)
    {
        assert(Shape.rank > 1, "can only transpose shapes greater than rank 1")

        func makePositive(dims: Shape) -> Shape {
            var positive = dims
            for i in 0..<Shape.rank where positive[i] < 0 {
                positive[i] += Shape.rank
            }
            return positive
        }

        // determine the new bounds and strides
        var shape = other.shape
        var strides = other.strides
        if let perm = permutations {
            let mapping = makePositive(dims: perm)
            for index in 0..<Shape.rank {
                shape[index] = other.shape[mapping[index]]
                strides[index] = other.strides[mapping[index]]
            }
        } else {
            // simple swap of last two dimensions
            shape.swapAt(Shape.rank-1, Shape.rank-2)
            strides.swapAt(Shape.rank-1, Shape.rank-2)
        }

        //-----------------------------------
        self.init(shape: shape,
                  strides: strides,
                  count: other.count,
                  spanCount: other.count,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: strides.areSequential(for: shape))
    }
    
    /// - Returns: transpose of self
    @differentiable(where Element: DifferentiableElement)
    @inlinable var t: Self { Self(transposing: self) }
    
    @differentiable(where Element: DifferentiableElement)
    @inlinable func transposed(permutatedBy permutations: Shape.Tuple) -> Self {
        Self(transposing: self, permutatedBy: Shape(permutations))
    }
}

extension Tensor where Element: DifferentiableElement {
    
    @derivative(of: init(transposing:permutatedBy:))
    @inlinable static func _vjpInit(
        transposing other: Self,
        permutatedBy permutations: Shape?
    ) -> (value: Self, pullback: (Self) -> Self)
    {
        let value = Self(transposing: other, permutatedBy: permutations)
        return (value, {
            Self(shape: other.shape,
                 strides: other.strides,
                 count: other.count,
                 spanCount: other.count,
                 storage: $0.storage,
                 baseOffset: $0.baseOffset,
                 order: $0.storageOrder,
                 share: $0.isShared,
                 isSequential: other.isSequential)
        })
    }
}

//==============================================================================
//
extension Tensor where Element: Numeric {
    //--------------------------------------------------------------------------
    /// init(zeros shape:order:
    /// creates a dense shape filled with zeros
    /// - Parameters:
    ///  - shape: the n-dimensional shape of the tensor to be filled
    ///  - order: the storage order of the elements
    @inlinable public init(zeros shape: Shape, order: StorageOrder = .C) {
        self.init(shape, order: order)
        fill(&self, with: 0)
    }

    @inlinable public init(zeros shape: Shape.Tuple, order: StorageOrder = .C) {
        self.init(zeros: Shape(shape), order: order)
    }
    
    //--------------------------------------------------------------------------
    /// init(ones shape:order:
    /// creates a dense shape filled with ones
    /// - Parameters:
    ///  - shape: the n-dimensional shape of the tensor to be filled
    ///  - order: the storage order of the elements
    @inlinable public init(ones shape: Shape, order: StorageOrder = .C) {
        self.init(shape, order: order)
        fill(&self, with: 1)
    }

    @inlinable public init(ones shape: Shape.Tuple, order: StorageOrder = .C) {
        self.init(ones: Shape(shape), order: order)
    }

    //--------------------------------------------------------------------------
    /// init(eye:offset:
    /// Returns a new data shape where the bounds and strides are permuted
    /// - Parameters:
    ///  - shape: the shape of the array
    ///  - offset: the offset of the diagonal
    ///  - order: the storage order of the new tensor
    @inlinable public init(
        eye shape: Shape,
        offset: Int = 0,
        order: StorageOrder = .C
    ) {
        self.init(shape, order: order)
        Context.currentQueue.eye(&self, offset: offset)
    }

    @inlinable public init(
        eye shape: Shape.Tuple,
        offset: Int = 0,
        order: StorageOrder = .C
    ) {
        self.init(eye: Shape(shape), offset: offset, order: order)
    }
}

//==============================================================================
// casting for convertible types
public extension Tensor {
    //--------------------------------------------------------------------------
    /// casting
    /// - Parameter other: a tensor of the same shape whose elements are
    /// to be cast
    @inlinable init<E>(_ other: Tensor<Shape,E>)
    where Element: BinaryFloatingPoint, E: BinaryInteger
    {
        self = cast(other)
    }

    @inlinable init<E>(_ other: Tensor<Shape,E>)
    where Element: BinaryInteger, E: BinaryFloatingPoint
    {
        self = cast(other)
    }
}

