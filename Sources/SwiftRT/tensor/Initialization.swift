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
    /// - Parameter other: a tensor of the same shape whose elements are
    /// to be cast
    @inlinable
    init<U>(_ other: U) where
        U: TensorView, U.Element: AnyConvertable, U.Shape == Self.Shape
    {
        self = Platform.service.cast(other)
    }
}

//==============================================================================
//
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
    init(concatenating tensors: Self..., alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = Self(concatenating: tensors, alongAxis: axis, name: name)
    }
    
    @inlinable
    init(concatenating tensors: [Self], alongAxis axis: Int = 0,
         name: String? = nil)
    {
        self = Platform.service.concat(tensors, alongAxis: axis, name)
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
        var stacked = Self.create(Shape(bounds: stackedExtents), nil)
        
        // copy others into place
        var lower = Bounds.zero
        for tensor in expanded {
            var view = stacked.sharedView(from: lower, bounds: tensor.bounds)
            Platform.service.copy(from: tensor, to: &view)
            lower[axis] += 1
        }
        self = stacked
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init(repeating value: Element, to bounds: Bounds, name: String? = nil)
    {
        let shape = Shape(bounds: bounds, strides: Bounds.zero)
        self = Self.create(for: value, shape, name)
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init(repeating value: Element, to bounds: BoundsTuple, name: String? = nil)
    {
        self.init(repeating: value, to: Bounds(bounds), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    init<U>(repeating value: Element, like other: U, name: String? = nil)
        where U: TensorView, Self.Shape == U.Shape
    {
        self = Self(repeating: value, to: other.bounds, name: name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(shape:
    @inlinable
    func createDense(with shape: Shape, name: String? = nil) -> Self {
        Self.create(shape.dense, name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense(bounds:
    @inlinable
    func createDense(with bounds: Bounds, name: String? = nil) -> Self {
        Self.create(Shape(bounds: bounds), name)
    }
    
    //--------------------------------------------------------------------------
    /// createDense()
    @inlinable
    func createDense() -> Self { return createDense(with: shape) }
    
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
    func createSingleElement(name: String? = nil) -> Self {
        Self.create(Shape(bounds: Bounds.one, strides: Bounds.one), name)
    }
    
    //==========================================================================
    // utility functions for creating shaped types
    @inlinable
    static func create(_ shape: Shape, _ name: String?) -> Self {
        let label = name ?? Self.diagnosticName
        let buffer = Buffer(count: shape.count, name: label)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeBufferPointer<Element>,
                       _ shape: Shape, _ name: String?) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let label = name ?? Self.diagnosticName
        let reference = Buffer(referenceTo: buffer, name: label)
        return Self(shape: shape, buffer: reference, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                       _ shape: Shape, _ name: String?) -> Self {
        assert(shape.count == buffer.count,
               "shape count does not match buffer count")
        // create tensor data reference to buffer
        let label = name ?? Self.diagnosticName
        let reference = Buffer(referenceTo: buffer, name: label)
        return Self(shape: shape, buffer: reference, offset: 0, shared: false)
    }
    
    @inlinable
    static func create(for element: Element, _ shape: Shape, _ name: String?) -> Self
    {
        let buffer = Buffer(for: element, name: name ?? Self.diagnosticName)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }

    @inlinable
    static func create<C>(_ elements: C, _ shape: Shape, _ name: String?) -> Self
        where C: Collection, C.Element == Element
    {
        // it can be less if the elements are being repeated
        assert(elements.count <= shape.count, _messageElementCountMismatch)
        let label = name ?? Self.diagnosticName

        // create the tensor
        let buffer = Buffer(elements: elements, name: label)
        return Self(shape: shape, buffer: buffer, offset: 0, shared: false)
    }
}

//==============================================================================
//
public extension TensorView where Self: DifferentiableTensorView {
    @inlinable
    @derivative(of: init(repeating:to:name:))
    static func _vjpInit(repeating value: Element,
                         to bounds: Bounds,
                         name: String?) ->
        (value: Self, pullback: (Self) -> (Element))
    {
        (Self(repeating: value, to: bounds), { $0.sum().element })
    }
}

