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

//==============================================================================
// error messages
public let _messageElementCountMismatch =
"the number of initial elements must equal the tensor size"

public let _messageNewTensorsShouldBeDense = "new tensors should be dense"

//==============================================================================
// casting for convertable types
public extension TensorView where Element: AnyConvertable {
    //--------------------------------------------------------------------------
    /// casting
    @inlinable
    init<U>(_ other: U) where
        U: TensorView, U.Element: AnyConvertable, Shape == U.Shape
    {
        self = cast(other)
    }
}

public typealias RangeInterval = (from: Int?, to: Int?, step: Int?)
public typealias ResolvedRangeInterval = (from: Int, to: Int, step: Int)

//==============================================================================
//
public extension TensorView {
    //--------------------------------------------------------------------------
    /// empty
    /// creates an empty tensor that can be used where a return
    /// value is needed in an error condition.
    @inlinable
    init() {
        self.init(shape: Shape(extents: Shape.zeros),
                  tensorArray: TensorArray(),
                  viewOffset: 0,
                  isMutable: false)
    }

    //--------------------------------------------------------------------------
    /// creates a tensor of the same type and shape as `self` with `Element`
    /// equal to `Bool`
    @inlinable
    func createBoolTensor() -> BoolView { createBoolTensor(with: extents) }
    /// creates a tensor of the same shape as `self` with `Element`
    /// equal to `IndexType`
    @inlinable
    func createIndexTensor() -> IndexView { createIndexTensor(with: extents) }

    //--------------------------------------------------------------------------
    /// concatenated tensors
    @inlinable
    init(concatenating tensors: Self..., alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = Self(concatenating: tensors, alongAxis: axis, name: name)
    }
    
    @inlinable
    init(concatenating tensors: [Self], alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = SwiftRT.concat(tensors: tensors, alongAxis: axis, name: name)
    }
    
    //--------------------------------------------------------------------------
    // flattening
    @inlinable
    init<T>(flattening other: T) where T: TensorView, T.Element == Element {
        self.init(shape: Shape(flattening: other.shape),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isMutable: other.isMutable)
    }

    // noop flattening case
    // this might be used when blindly flattening an input to
    // a dense matmul expression
    @inlinable
    init(flattening other: Self) {
        self = other
    }
    
    //--------------------
    // derivative
    @inlinable
    @derivative(of: init(flattening:))
    static func _vjpInit<T>(flattening other: T) ->
        (value: Self, pullback: (Self) -> T.TangentVector) where
        Self: DifferentiableTensorView,
        T: DifferentiableTensorView, T.Element == Element
    {
        let value = Self(flattening: other)
        let rank = Shape.zeros.count
        let axes = Set([Int](rank..<other.rank))
        return (value, { T(expanding: $0, alongAxes: axes) })
    }
    
    //--------------------------------------------------------------------------
    // init(indenting:
    @inlinable
    init<T>(indenting other: T) where T: TensorView, T.Element == Element {
        self.init(shape: Shape(indenting: other.shape),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isMutable: other.isMutable)
    }
        
    //--------------------------------------------------------------------------
    // init(padding:
    @inlinable
    init<T>(padding other: T) where T: TensorView, T.Element == Element {
        self.init(shape: Shape(padding: other.shape),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isMutable: other.isMutable)
    }
    
    //--------------------------------------------------------------------------
    // expanding
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(expanding other: T, alongAxes axes: Set<Int>? = nil)
        where T: TensorView, T.Element == Element
    {
        self.init(shape: Shape(expanding: other.shape, alongAxes: axes),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isMutable: other.isMutable)
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(expanding other: T, alongAxes axes: Int...)
        where T: TensorView, T.Element == Element {
            self.init(expanding: other, alongAxes: Set(axes))
    }

    //--------------------
    // derivative
    @inlinable
    @derivative(of: init(expanding:alongAxes:))
    static func _vjpInit<T>(expanding other: T, alongAxes axes: Set<Int>?) ->
        (value: Self, pullback: (Self) -> T.TangentVector) where
        Self: DifferentiableTensorView,
        T: DifferentiableTensorView, T.Element == Element
    {
        let value = Self(expanding: other, alongAxes: axes)
        return (value, { T(squeezing: $0, alongAxes: axes) })
    }

    //--------------------------------------------------------------------------
    // squeezing
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(squeezing other: T, alongAxes axes: Set<Int>? = nil)
        where T: TensorView, T.Element == Element
    {
        self.init(shape: Shape(squeezing: other.shape, alongAxes: axes),
                  tensorArray: other.tensorArray,
                  viewOffset: other.viewOffset,
                  isMutable: other.isMutable)
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(squeezing other: T, alongAxes axes: Int...)
        where T: TensorView, T.Element == Element {
        self.init(squeezing: other, alongAxes: Set(axes))
    }

    //--------------------
    // derivative
    @inlinable
    @derivative(of: init(squeezing:alongAxes:))
    static func _vjpInit<T>(squeezing other: T, alongAxes axes: Set<Int>?) ->
        (value: Self, pullback: (Self) -> T.TangentVector)
        where Self: DifferentiableTensorView,
        T: DifferentiableTensorView, T.Element == Element
    {
        let value = Self(squeezing: other, alongAxes: axes)
        return (value, { T(expanding: $0, alongAxes: axes) })
    }
    
    //--------------------------------------------------------------------------
    // stacking
    @inlinable
    init<T>(stacking others: T..., alongAxis axis: Int = 0)
        where T: TensorView, T.Element == Element
    {
        self.init(stacking: others, alongAxis: axis)
    }
    
    @inlinable
    init<T>(stacking others: [T], alongAxis axis: Int = 0)
        where T: TensorView, T.Element == Element
    {
        // verify that tensors are the correct rank and same shape
        let rank = Shape.zeros.count
        assert(others.count > 0 && others[0].rank == rank - 1,
               "stacked tensors must be of rank \(rank - 1)")
        assert({
            let extents = others[0].extents
            for i in 1..<others.count {
                if others[i].extents != extents { return false }
            }
            return true
        }(), "stacked tensors must all be the same size")

        // form stacked extents and create dense stacked result
        let expanded = others.map { Self(expanding: $0, alongAxes: axis) }
        var stackedExtents = expanded[0].extents
        stackedExtents[axis] = expanded.count
        var stacked = Self.create(Shape(extents: stackedExtents), nil)

        // copy others into place
        var index = Shape.zeros
        for tensor in expanded {
            var view = stacked.mutableView(at: index, extents: tensor.extents)
            copy(from: tensor, to: &view)
            index[axis] += 1
        }
        self = stacked
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init(repeating value: Element, to extents: Shape.Array, name: String? = nil)
    {
        let shape = Shape(extents: extents, strides: Shape.zeros)
        self = Self.create([value], shape, name)
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init(repeating value: Element, to extents: Shape.Tuple, name: String? = nil)
    {
        self.init(repeating: value, to: Shape.Array(extents), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init<U>(repeating value: Element, like other: U, name: String? = nil)
        where U: TensorView, Self.Shape == U.Shape
    {
        self = Self(repeating: value, to: other.extents, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(shape:
    @inlinable
    func createDense(with shape: Shape, name: String? = nil) -> Self {
        Self.create(shape.dense, name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(extents:
    @inlinable
    func createDense(with extents: Shape.Array, name: String? = nil) -> Self {
        let newShape = isContiguous ?
            Shape(extents: extents, strides: self.shape.strides) :
            Shape(extents: extents)
        return createDense(with: newShape, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense()
    @inlinable
    func createDense() -> Self { return createDense(with: shape) }
    
    //--------------------------------------------------------------------------
    /// reductionExtents
    /// determines the extents of a reduction result along the specified axes
    @inlinable
    func reductionExtents(alongAxes axes: Set<Int>?) -> Shape.Array {
        guard let axes = axes else { return Shape.ones }
        assert(axes.isSubset(of: 0..<rank), "axis is out of bounds")
        var resultExtents = extents
        axes.forEach { resultExtents[$0] = 1 }
        return resultExtents
    }

    //--------------------------------------------------------------------------
    /// createSingleElement
    /// helper to create a rank extended value
    @inlinable
    func createSingleElement(name: String? = nil) -> Self {
        Self.create(Shape(extents: Shape.ones, strides: Shape.ones), name)
    }
    
    //==========================================================================
    // utility functions for creating shaped types
    @inlinable
    static func create(_ shape: Shape, _ name: String?) -> Self {
        let label = name ?? Self.diagnosticName
        let array = TensorArray<Element>(count: shape.count, name: label)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isMutable: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeBufferPointer<Element>,
                       _ shape: Shape, _ name: String?) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let label = name ?? Self.diagnosticName
        let array = TensorArray<Element>(referenceTo: buffer, name: label)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isMutable: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                       _ shape: Shape, _ name: String?) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let label = name ?? Self.diagnosticName
        let array = TensorArray<Element>(referenceTo: buffer, name: label)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isMutable: false)
    }
    
    @inlinable
    static func create<C>(_ elements: C, _ shape: Shape,
                          _ name: String?) -> Self where
        C: Collection, C.Element == Element
    {
        // it can be less if the elements are being repeated
        assert(elements.count <= shape.count, _messageElementCountMismatch)
        let label = name ?? Self.diagnosticName
        let array = TensorArray<Element>(elements: elements, name: label)
        return Self(shape: shape, tensorArray: array,
                    viewOffset: 0, isMutable: false)
    }
}

//==============================================================================
//
//public extension TensorView where Self: DifferentiableTensorView {
//    @inlinable
//    @derivative(of: init(repeating:to:name:))
//    static func _vjpInit(repeating value: Element, to extents: Shape.Array,
//                         name: String?) ->
//        (value: Self, pullback: (Self) -> (Element))
//    {
//        (Self(repeating: value, to: extents), { $0.sum().element })
//    }
//}

