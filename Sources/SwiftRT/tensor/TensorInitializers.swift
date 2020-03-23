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
// casting for convertible types
public extension TensorView {
    //--------------------------------------------------------------------------
    /// casting
    /// - Parameter other: a tensor of the same shape whose elements are
    /// to be cast
    @inlinable
    init<U>(_ other: U) where
        Self.Element: BinaryFloatingPoint,
        U: TensorView, U.Element: BinaryInteger, U.Bounds == Bounds
    {
        self = Context.platform.cast(other)
    }

    @inlinable
    init<U>(_ other: U) where
        Self.Element: BinaryInteger,
        U: TensorView, U.Element: BinaryFloatingPoint, U.Bounds == Bounds
    {
        self = Context.platform.cast(other)
    }
}

//==============================================================================
// basic initializers
public extension TensorView {
    //--------------------------------------------------------------------------
    /// creates a tensor of the same type and shape as `self` with `Element`
    /// equal to `Bool`
    @inlinable
    func createBoolTensor() -> BoolView { createBoolTensor(with: bounds) }
    /// creates a tensor of the same shape as `self` with `Element`
    /// equal to `IndexType`
    @inlinable
    func createIndexTensor() -> IndexView { createIndexTensor(with: bounds) }

    //--------------------------------------------------------------------------
    /// concatenated tensors
    @inlinable
    init(concatenating tensors: Self..., alongAxis axis: Int = 0) {
        self = Self(concatenating: tensors, alongAxis: axis)
    }
    
    @inlinable
    init(concatenating tensors: [Self], alongAxis axis: Int = 0) {
        self = Context.platform.concat(tensors, alongAxis: axis)
    }
    
    //--------------------------------------------------------------------------
    // flattening
    @inlinable
    init<T>(flattening other: T) where
        T: TensorView, T.Buffer == Buffer
    {
        self.init(shape: Shape(flattening: other.shape),
                  buffer: other.buffer,
                  offset: other.offset,
                  shared: other.shared)
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
        T: DifferentiableTensorView, T.Buffer == Buffer
    {
        let value = Self(flattening: other)
        let axes = Set([Int](Self.rank..<T.rank))
        return (value, { T(expanding: $0, alongAxes: axes) })
    }
    
    //--------------------------------------------------------------------------
    // init(indenting:
    @inlinable
    init<T>(indenting other: T) where
        T: TensorView, T.Buffer == Buffer
    {
        self.init(shape: Shape(indenting: other.shape),
                  buffer: other.buffer,
                  offset: other.offset,
                  shared: other.shared)
    }
        
    //--------------------------------------------------------------------------
    // init(padding:
    @inlinable
    init<T>(padding other: T) where
        T: TensorView, T.Buffer == Buffer
    {
        self.init(shape: Shape(padding: other.shape),
                  buffer: other.buffer,
                  offset: other.offset,
                  shared: other.shared)
    }
    
    //--------------------------------------------------------------------------
    // expanding
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(expanding other: T, alongAxes axes: Set<Int>? = nil)
        where T: TensorView, T.Buffer == Buffer
    {
        self.init(shape: Shape(expanding: other.shape, alongAxes: axes),
                  buffer: other.buffer,
                  offset: other.offset,
                  shared: other.shared)
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(expanding other: T, alongAxes axes: Int...)
        where T: TensorView, T.Buffer == Buffer {
            self.init(expanding: other, alongAxes: Set(axes))
    }

    //--------------------
    // derivative
    @inlinable
    @derivative(of: init(expanding:alongAxes:))
    static func _vjpInit<T>(expanding other: T, alongAxes axes: Set<Int>?) ->
        (value: Self, pullback: (Self) -> T.TangentVector) where
        Self: DifferentiableTensorView,
        T: DifferentiableTensorView, T.Buffer == Buffer
    {
        let value = Self(expanding: other, alongAxes: axes)
        return (value, { T(squeezing: $0, alongAxes: axes) })
    }

    //--------------------------------------------------------------------------
    // squeezing
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(squeezing other: T, alongAxes axes: Set<Int>? = nil)
        where T: TensorView, T.Buffer == Buffer
    {
        self.init(shape: Shape(squeezing: other.shape, alongAxes: axes),
                  buffer: other.buffer,
                  offset: other.offset,
                  shared: other.shared)
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView, T: DifferentiableTensorView)
    init<T>(squeezing other: T, alongAxes axes: Int...)
        where T: TensorView, T.Buffer == Buffer {
        self.init(squeezing: other, alongAxes: Set(axes))
    }

    //--------------------
    // derivative
    @inlinable
    @derivative(of: init(squeezing:alongAxes:))
    static func _vjpInit<T>(squeezing other: T, alongAxes axes: Set<Int>?) ->
        (value: Self, pullback: (Self) -> T.TangentVector)
        where Self: DifferentiableTensorView,
        T: DifferentiableTensorView, T.Buffer == Buffer
    {
        let value = Self(squeezing: other, alongAxes: axes)
        return (value, { T(expanding: $0, alongAxes: axes) })
    }
    
    //--------------------------------------------------------------------------
    // stacking
    @inlinable
    init<T>(stacking others: T..., alongAxis axis: Int = 0)
        where T: TensorView, T.Buffer == Buffer
    {
        self.init(stacking: others, alongAxis: axis)
    }
    
    @inlinable
    init<T>(stacking others: [T], alongAxis axis: Int = 0)
        where T: TensorView, T.Buffer == Buffer
    {
        // verify that tensors are the correct rank and same shape
        assert(others.count > 0 && T.rank == Self.rank - 1,
               "stacked tensors must be of rank \(Self.rank - 1)")
        assert({
            let bounds = others[0].bounds
            for i in 1..<others.count {
                if others[i].bounds != bounds { return false }
            }
            return true
        }(), "stacked tensors must all be the same size")
        
        // form stacked bounds and create dense stacked result
        let expanded = others.map { Self(expanding: $0, alongAxes: axis) }
        var stackedExtents = expanded[0].bounds
        stackedExtents[axis] = expanded.count
        var stacked = Self.create(Shape(stackedExtents))
        
        // copy others into place
        var lower = Bounds.zero
        for tensor in expanded {
            var view = stacked.sharedView(from: lower, bounds: tensor.bounds)
            Context.platform.copy(from: tensor, to: &view)
            lower[axis] += 1
        }
        self = stacked
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init(repeating value: Element, to bounds: Bounds) {
        self = Self.create(for: value, Shape(bounds, strides: Bounds.zero))
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init<U>(repeating value: Element, like other: U)
        where U: TensorView, Self.Bounds == U.Bounds
    {
        self = Self(repeating: value, to: other.bounds)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(shape:
    @inlinable
    func createDense(with shape: Shape<Bounds>) -> Self {
        Self.create(shape.dense)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(bounds:
    @inlinable
    func createDense(with bounds: Bounds) -> Self {
        Self.create(Shape(bounds))
    }
    
    //--------------------------------------------------------------------------
    /// createDense()
    @inlinable
    func createDense() -> Self { createDense(with: shape) }
    
    //--------------------------------------------------------------------------
    /// reductionBounds
    /// returns the upper bounds for a reduction result along the specified axes
    @inlinable
    func reductionBounds(alongAxes axes: Set<Int>?) -> Bounds {
        guard let axes = axes else { return Bounds.one }
        assert(axes.isSubset(of: 0..<Self.rank), "axis is out of bounds")
        var result = bounds
        axes.forEach { result[$0] = 1 }
        return result
    }

    //--------------------------------------------------------------------------
    /// createSingleElement
    /// helper to create a rank extended value
    @inlinable
    func createSingleElement() -> Self {
        Self.create(Shape(Bounds.one, strides: Bounds.one))
    }

    //==========================================================================
    // utility functions for creating shaped types
    @inlinable
    static func create(_ shape: Shape<Bounds>) -> Self {
        let buffer = Buffer(count: shape.count, name: Self.diagnosticName)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeBufferPointer<Element>,
                       _ shape: Shape<Bounds>) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let reference = Buffer(referenceTo: buffer, name: Self.diagnosticName)
        return Self(shape: shape, buffer: reference, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                       _ shape: Shape<Bounds>) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let reference = Buffer(referenceTo: buffer, name: Self.diagnosticName)
        return Self(shape: shape, buffer: reference, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(for element: Element, _ shape: Shape<Bounds>) -> Self
    {
        let buffer = Buffer(for: element, name: Self.diagnosticName)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }

    @inlinable
    static func create<C>(_ elements: C, _ shape: Shape<Bounds>) -> Self
        where C: Collection, C.Element == Element
    {
        // it can be less if the elements are being repeated
        assert(elements.count <= shape.count, _messageElementCountMismatch)
        let buffer = Buffer(elements: elements, name: Self.diagnosticName)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }
}

//==============================================================================
// simple numeric helpers
public extension TensorView where Element: Numeric {
    @inlinable init(zeros bounds: Bounds) {
        self.init(repeating: 0, to: bounds)
    }
    
    @inlinable init(zeros bounds: Bounds.Tuple) {
        self.init(repeating: 0, to: Bounds(bounds))
    }

    @inlinable init(ones bounds: Bounds) {
        self.init(repeating: 1, to: bounds)
    }

    @inlinable init(ones bounds: Bounds.Tuple) {
        self.init(repeating: 1, to: Bounds(bounds))
    }

    @inlinable init<U>(zerosLike other: U)
        where U: TensorView, U.Bounds == Bounds
    {
        self.init(repeating: 0, to: other.bounds)
    }

    @inlinable init<U>(onesLike other: U)
        where U: TensorView, U.Bounds == Bounds
    {
        self.init(repeating: 1, to: other.bounds)
    }
}

//==============================================================================
//
public extension TensorView where Self: DifferentiableTensorView {
    @inlinable
    @derivative(of: init(repeating:to:))
    static func _vjpInit(repeating value: Element, to bounds: Bounds)
        -> (value: Self, pullback: (Self) -> (Element))
    {
        (Self(repeating: value, to: bounds), { $0.sum().element })
    }
}

