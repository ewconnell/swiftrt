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
                  strides: shape.sequentialStrides(),
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
    @inlinable init(repeating element: Element, to shape: Shape) {
        self.init(constant: element, shape: shape)
    }

    //--------------------------------------------------------------------------
    /// init(repeating other:shape:
    /// Repeats a tensor withing the specified shape while indexing
    /// - Parameters:
    ///  - other: the tensor to repeat
    ///  - shape: the shape of the tensor
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
    /// init(flattening:
    /// - Parameter other: the shape to flatten
    @inlinable init<S>(flattening other: Tensor<S,Element>)
        where S: TensorShape
    {
        // TODO: consider special cases where this restriction might be lifted
        assert(other.isSequential, "cannot flatten non sequential data")
        let shape = Shape(flattening: other.shape)
        self.init(shape: shape,
                  strides: shape.sequentialStrides(),
                  elementCount: other.elementCount,
                  spanCount: other.elementCount,
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: true)
    }

    // noop flattening case
    // this might be used when blindly flattening a parameter
    @inlinable init(flattening other: Self) {
        self = other
    }

    //--------------------------------------------------------------------------
    /// init(expanding:
    /// - Parameter other: the shape to expand
    @inlinable init<S>(
        expanding other: Tensor<S,Element>,
        alongAxes axes: Set<Int>? = nil
    ) where S: TensorShape
    {
        //-----------------------------------
        assert(S.rank < Shape.rank, "can only expand lower ranked shapes")
        var shape = Shape.zero
        var strides = Shape.zero
        let axesSet = axes == nil ?
            Set(0..<Shape.rank - S.rank) :
            Set(axes!.map { $0 < 0 ? $0 + Shape.rank : $0 })
        assert(S.rank + axesSet.count == Shape.rank,
               "`other.rank` plus number of specified axes " +
            "must equal the `rank` of this shape")

        var j = S.rank - 1
        for i in (0..<Shape.rank).reversed() {
            if axesSet.contains(i) {
                // expanded axes are set to 1
                shape[i] = 1
                // repeat stride of next dimension or pad with 1
                if i == Shape.rank - 1 {
                    strides[i] = 1
                } else {
                    strides[i] = shape[i + 1] * strides[i + 1]
                }
            } else {
                shape[i] = other.shape[j]
                strides[i] = other.strides[j]
                j -= 1
            }
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
    
    @inlinable
    init<S>(expanding other: Tensor<S,Element>, alongAxes axes: Int...)
        where S: TensorShape
    {
        self.init(expanding: other, alongAxes: Set(axes))
    }

    //--------------------------------------------------------------------------
    /// init(squeezing:
    /// - Parameter other: the shape to expand
    @inlinable init<S>(
        squeezing other: Tensor<S,Element>,
        alongAxes axes: Set<Int>? = nil
    ) where S: TensorShape
    {
        //-----------------------------------
        // make sure we have a positive set of axes to squeeze along
        var shape = Shape.zero
        var strides = Shape.zero
        let axesSet = axes == nil ?
            Set(0..<S.rank) :
            Set(axes!.map { $0 < 0 ? S.rank + $0 : $0 })
        
        var axis = 0
        for otherAxis in 0..<S.rank where
            !(other.shape[otherAxis] == 1 && axesSet.contains(otherAxis))
        {
            assert(axis < Shape.rank,
                   "Unsqueezed axes of `other` exceeds rank of this shape")
            shape[axis] = other.shape[otherAxis]
            strides[axis] = other.strides[otherAxis]
            axis += 1
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
    
    @inlinable
    init<S>(squeezing other: Tensor<S,Element>, alongAxes axes: Int...)
        where S: TensorShape
    {
        self.init(squeezing: other, alongAxes: Set(axes))
    }

    //--------------------------------------------------------------------------
    /// init(stacking:
    @inlinable init<S>(
        stacking others: [Tensor<S,Element>],
        alongAxis axis: Int = 0
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
        let expanded = others.map { Self(expanding: $0, alongAxes: axis) }
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
    
    @inlinable
    init<S>(stacking others: Tensor<S,Element>..., alongAxis axis: Int = 0) {
        self.init(stacking: others, alongAxis: axis)
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
    @inlinable init(transposing other: Self, with permutations: Shape? = nil) {
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
                  isSequential: strides == shape.sequentialStrides())
    }
    
    /// - Returns: transpose of self
    @inlinable var t: Self { Self(transposing: self) }
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
}

//==============================================================================
// initializer derivatives
extension Tensor where Element: DifferentiableElement
{
    //--------------------------------------------------------------------------
    // TODO: THIS IS REALLY EXPENSIVE AND USELESS!!!!
    @derivative(of: init(repeating:to:))
    @inlinable static func _vjpInit(repeating value: Element, to shape: Shape)
        -> (value: Self, pullback: (Self) -> (Element))
    {
        (Self(repeating: value, to: shape), { $0.sum().element })
    }
    
    //--------------------------------------------------------------------------
    @derivative(of: init(flattening:))
    @inlinable static func _vjpInit<S>(flattening other: Tensor<S,Element>)
        -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape
    {
        let axes = Set([Int](Shape.rank..<S.rank))
        let value = Self(flattening: other)
        return (value, { Tensor<S,Element>(expanding: $0, alongAxes: axes) })
    }

    //--------------------------------------------------------------------------
    @derivative(of: init(expanding:alongAxes:))
    @inlinable static func _vjpInit<S>(
        expanding other: Tensor<S,Element>,
        alongAxes axes: Set<Int>?
    ) -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape
    {
        let value = Self(expanding: other, alongAxes: axes)
        return (value, { Tensor<S,Element>(squeezing: $0, alongAxes: axes) })
    }

    //--------------------------------------------------------------------------
    @derivative(of: init(squeezing:alongAxes:))
    @inlinable static func _vjpInit<S>(
        squeezing other: Tensor<S,Element>,
        alongAxes axes: Set<Int>?
    ) -> (value: Self, pullback: (Self) -> Tensor<S,Element>)
        where S: TensorShape
    {
        let value = Self(squeezing: other, alongAxes: axes)
        return (value, { Tensor<S,Element>(expanding: $0, alongAxes: axes) })
    }

}
