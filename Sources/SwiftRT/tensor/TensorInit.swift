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
public typealias Tensor1<Element> = Tensor<Shape1, Element>
public typealias Tensor2<Element> = Tensor<Shape2, Element>
public typealias Tensor3<Element> = Tensor<Shape3, Element>
public typealias Tensor4<Element> = Tensor<Shape4, Element>
public typealias Tensor5<Element> = Tensor<Shape5, Element>
public typealias Tensor6<Element> = Tensor<Shape6, Element>

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
public extension Tensor {
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
                  elementCount: count,
                  spanCount: count,
                  storage: StorageBufferType(count: count, name: Self.name),
                  baseOffset: 0,
                  order: order,
                  share: false,
                  isSequential: true)
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
        self.init(shape: Shape.one,
                  strides: Shape.one,
                  elementCount: 1,
                  spanCount: 1,
                  storage: StorageBufferType(count: 1, name: Self.name),
                  baseOffset: 0,
                  order: .C,
                  share: false,
                  isSequential: true)
        storage.setElement(value: element, at: 0)
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
        let elementCount = shape.elementCount()
        self.init(shape: shape,
                  strides: repeatedStrides,
                  elementCount: elementCount,
                  spanCount: shape.spanCount(stridedBy: repeatedStrides),
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: elementCount == 1)
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
        _ = storage.hostBuffer.initialize(from: elements)
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
        _ = storage.hostBuffer.initialize(from: lazyElements)
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
        _ = storage.hostBuffer.initialize(from: lazyElements)
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
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// reductionBounds
    /// returns the upper bounds for a reduction result along the specified axes
    @inlinable func reductionShape(alongAxes axes: Set<Int>?) -> Shape {
        guard let axes = axes else { return Shape.one }
        assert(axes.isSubset(of: 0..<Shape.rank), "axis is out of bounds")
        var result = shape
        axes.forEach { result[$0] = 1 }
        return result
    }
}

//==============================================================================
// Rank transformations
public extension Tensor {
    //--------------------------------------------------------------------------
    /// concatenated tensors
    @inlinable init(concatenating tensors: Self..., alongAxis axis: Int = 0) {
        self = Self(concatenating: tensors, alongAxis: axis)
    }
    
    @inlinable init(concatenating tensors: [Self], alongAxis axis: Int = 0) {
        self = SwiftRT.concat(tensors, alongAxis: axis)
    }

    //--------------------------------------------------------------------------
    /// init(reshaping:shape:order:
    /// - Parameters:
    ///  - other: the tensor to reshape
    ///  - newShape: the shape of the new tensor
    ///  - order: the storage order of the new tensor
    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(
        reshaping other: Tensor<S,Element>,
        to newShape: Shape,
        order newOrder: StorageOrder = .C
    ) where S: TensorShape
    {
        // TODO: consider special cases where this restriction can be lifted
        assert(other.isSequential, "cannot reshape non sequential data")
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
            assert(other.elementCount % specifiedCount == 0,
                   "incompatible dimensions")
            shape[i] = other.elementCount / specifiedCount
        }
        assert(shape.elementCount() == other.elementCount,
               "the new shape must have the same number of elements as other")
        
        // determine storage order
        let order: StorageOrder = newOrder == .F ||
            (newOrder == .A &&
            other.storageOrder == .F &&
            other.isSequential) ? .F : .C

        // create new tensor, which is a shaped reference to other's storage
        self.init(shape: shape,
                  strides: shape.strides(for: order),
                  elementCount: other.elementCount,
                  spanCount: other.spanCount,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: order,
                  share: other.isShared,
                  isSequential: true)

        // reorder elements if needed
        if order != other.storageOrder {
            // create other with the new shape retaining it's original order
            let reshapedOther =
                Self(shape: shape,
                     strides: shape.strides(for: other.storageOrder),
                     elementCount: other.elementCount,
                     spanCount: other.spanCount,
                     storage: other.storage,
                     baseOffset: other.baseOffset,
                     order: other.storageOrder,
                     share: other.isShared,
                     isSequential: true)
            
            // create a new empty storage buffer for self
            self.storage = StorageBufferType<Element>(count: elementCount,
                                                      name: storage.name)
            
            // performs an indexed copy which reorders the elements
            copy(from: reshapedOther, to: &self)
        }
    }

    //--------------------------------------------------------------------------
    /// init(expanding other:axes:
    /// - Parameters:
    ///  - other: the tensor to expand
    ///  - axes: the list of axes to expand
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
                  elementCount: other.elementCount,
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

    //--------------------------------------------------------------------------
    /// init(squeezing:
    /// - Parameters:
    ///  - other: the collection to squeeze
    ///  - axes: a list of axes to squeeze
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
        
        for i in 0..<S.rank where otherShape[i] == 0 {
            shape[axis] = otherShape[i]
            strides[axis] = other.strides[i]
            axis += 1
        }
        
        self.init(shape: shape,
                  strides: strides,
                  elementCount: other.elementCount,
                  spanCount: other.elementCount,
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

    //--------------------------------------------------------------------------
    /// init(stacking:axis:
//    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(
        stacking others: [Tensor<S,Element>],
        axis: Int = 0
    ) where S: TensorShape
    {
        // verify that tensors are the correct rank and same shape
        assert(others.count > 0 && S.rank == Shape.rank - 1,
               "stacked tensors must be of rank \(Shape.rank - 1)")
        assert({
            let shape = others[0].shape
            for i in 1..<others.count {
                if others[i].shape != shape { return false }
            }
            return true
        }(), "stacked tensors must all be the same size")
        
        // form stacked bounds and create dense stacked result
        let expanded = others.map { Self(expanding: $0, axes: Shape1(axis)) }
        var stackedShape = expanded[0].shape
        stackedShape[axis] = expanded.count
        self = Self(stackedShape)
        
        // copy others into place
        var lower = Shape.zero
        for tensor in expanded {
            self[lower, lower &+ tensor.shape] = tensor
            lower[axis] += 1
        }
    }
    
//    @differentiable(where Element: DifferentiableElement)
    @inlinable init<S>(stacking others: Tensor<S,Element>..., axis: Int = 0) {
        self.init(stacking: others, axis: axis)
    }

    //--------------------------------------------------------------------------
    /// init(indenting:
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
                  elementCount: other.elementCount,
                  spanCount: other.elementCount,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: other.isSequential)
    }
    
    //--------------------------------------------------------------------------
    /// init(transposing:permutations:
    /// Returns a new data shape where the bounds and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    /// - Precondition: Each value in `permutations` must be in the range
    ///   `-rank..<rank`
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
                  elementCount: other.elementCount,
                  spanCount: other.elementCount,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: strides.areSequential(for: shape))
    }
    
    /// - Returns: transpose of self
    @inlinable var t: Self { Self(transposing: self) }
    
    @inlinable func transposed(permutatedBy permutations: Shape.Tuple) -> Self {
        Self(transposing: self, permutatedBy: Shape(permutations))
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
    @inlinable init(zeros shape: Shape, order: StorageOrder = .C) {
        self.init(shape, order: order)
        fill(&self, with: 0)
    }

    //--------------------------------------------------------------------------
    /// init(ones shape:order:
    /// creates a dense shape filled with ones
    /// - Parameters:
    ///  - shape: the n-dimensional shape of the tensor to be filled
    ///  - order: the storage order of the elements
    @inlinable init(ones shape: Shape, order: StorageOrder = .C) {
        self.init(shape, order: order)
        fill(&self, with: 1)
    }
    
    //--------------------------------------------------------------------------
    /// init(eye:offset:
    /// Returns a new data shape where the bounds and strides are permuted
    /// - Parameters:
    ///  - shape: the shape of the array
    ///  - offset: the offset of the diagonal
    ///  - order: the storage order of the new tensor
    @inlinable init(
        eye shape: Shape,
        offset: Int = 0,
        order: StorageOrder = .C
    ) {
        let count = shape.elementCount()
        self.init(shape: shape,
                  strides: shape.strides(for: order),
                  elementCount: count,
                  spanCount: count,
                  storage: StorageBufferType(count: count, name: Self.name),
                  baseOffset: 0,
                  order: order,
                  share: false,
                  // set to `false` to force spatial indexing, which
                  // is needed by the driver function to generate the pattern
                  isSequential: false)

        Context.currentQueue.eye(&self, offset: offset)
    }
}

//==============================================================================
// initializer derivatives
extension Tensor where Element: DifferentiableElement
{
    //--------------------------------------------------------------------------
    @derivative(of: init(repeating:to:))
    @inlinable static func _vjpInit(repeating value: Element, to shape: Shape)
        -> (value: Self, pullback: (Self) -> (Element))
    {
        (Self(repeating: value, to: shape), { $0.sum().element })
    }
    
    //--------------------------------------------------------------------------
    @derivative(of: init(repeating:to:))
    @inlinable static func _vjpInit(repeating other: Self, to shape: Shape)
        -> (value: Self, pullback: (Self) -> (Self))
    {
        // TODO: this is probably wrong. Test this
        (Self(repeating: other, to: shape), { $0 })
    }
    
    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
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
