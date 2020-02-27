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

public extension TensorView where Self: VectorView {
    //--------------------------------------------------------------------------
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(index: Int) -> Element {
        get {
            view(at: makePositive(index: (index)),
                 extents: Shape.ones, strides: Shape.ones).element
        }
        set {
            expandSelfIfRepeated()
            var view = sharedView(at: makePositive(index: (index)),
                                  extents: Shape.ones, strides: Shape.ones)
            view.element = newValue
        }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
        {
        get {
            let r = range.relativeTo(0..<extents[0])
            return self[(r.start), (r.end), (r.step)]
        }
        set {
            let r = range.relativeTo(0..<extents[0])
            self[(r.start), (r.end), (r.step)] = newValue
        }
    }
}

public extension TensorView {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(range: UnboundedRange) -> Self { self }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get {
            let (start, end, steps) =
                getItemRange(range.relativeTo(0..<extents[0]))
            return self[start, end, steps]
        }
        set {
            let (start, end, steps) =
                getItemRange(range.relativeTo(0..<extents[0]))
            self[start, end, steps] = newValue
        }
    }
    
    @usableFromInline
    @_semantics("autodiff.nonvarying")
    internal func getItemRange(_ range: StridedRange<Int>) ->
        (Shape.Array, Shape.Array, Shape.Array)
    {
        var start = Shape.zeros
        var end = self.extents
        var steps = Shape.ones
        start[0] = range.start
        end[0] = range.end
        steps[0] = range.step
        return (start, end, steps)
    }
}

//==============================================================================
public extension TensorView {
    //--------------------------------------------------------------------------
    /// makeDense(view:
    /// if the view is already dense, then noop. If the view is repeated,
    /// then the view virtual elements are realized and the view is converted
    /// to dense.
    // Note: This is not part of mutableView, because there are cases
    // where we want to interact with a repeated view, such as in reductions
    @inlinable
    mutating func expandSelfIfRepeated() {
        guard spanCount < count else { return }

        // report
        diagnostic("\(expandingString) " +
            "\(name)(\(id)) storage: \(Element.self)[\(spanCount)] " +
            "expanded to: \(Element.self)[\(count)]",
            categories: [.dataCopy, .dataExpanding])

        // create storage for all elements
        var dense = createDense()
        copy(from: self, to: &dense)
        self = dense
    }

    //--------------------------------------------------------------------------
    /// getExtents(from:to:
    /// computes the extents and strides from the specified bounds and steps
    /// - Parameter lower: the lower bound of the subview
    /// - Parameter upper: the upper bound of the subview
    /// - Returns: the extents to be used to create a subview
    @inlinable
    func getExtents(from lower: Shape.Array, to upper: Shape.Array)
        -> Shape.Array
    {
        // bounds should be in the correct order by the time they reach here
        assert({
            for (l, u) in zip(lower, upper) { if l > u { return false } }
            return true
        }(), "lower must be less than or equal to upper")

        var extents = upper
        zip(extents.indices, zip(upper, lower)).forEach {
            extents[$0] = $1.0 - $1.1
        }
        return extents
    }

    //--------------------------------------------------------------------------
    /// getExtents(_:_:_
    /// computes the extents and strides from the specified bounds and steps
    /// - Parameter lower: the lower bound of the subview
    /// - Parameter upper: the upper bound of the subview
    /// - Parameter steps: the step interval along each dimension. This
    ///                    value can be negative to perform reverse traversal
    /// - Returns: the extents and strides to be used to create a subview
    @inlinable
    func getExtents(_ lower: Shape.Array,
                    _ upper: Shape.Array,
                    _ steps: Shape.Array) ->
        (extents: Shape.Array, strides: Shape.Array)
    {
        // if all the steps are 1, then just reuse the parent strides
        if steps.first(where: { $0 != 1 }) == nil {
            return (getExtents(from: lower, to: upper), self.strides)

        } else {
            // if one or more steps are not 1,
            // then recompute the subview extents and strides

            // y must be positive for this to work correctly
            func divceil(_ x: Int, _ y: Int) -> Int { (x - 1 + y) / y }
            
            var subExtents = getExtents(from: lower, to: upper)
            zip(subExtents.indices, zip(subExtents, steps)).forEach {
                subExtents[$0] = divceil($1.0, Swift.abs($1.1))
            }
            var subStrides = strides
            zip(subStrides.indices, zip(strides, steps)).forEach {
                subStrides[$0] = $1.0 * $1.1
            }
            return (subExtents, subStrides)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(lower: Shape.Tuple, upper: Shape.Tuple, steps: Shape.Tuple)
        -> Self {
        get { self[Shape.Array(lower), Shape.Array(upper), Shape.Array(steps)] }
        set {
            self[Shape.Array(lower), Shape.Array(upper),
                 Shape.Array(steps)] = newValue
        }
    }
    
    //--------------------------------------------------------------------------
    // views will have the same shared state as the parent
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(lower: Shape.Array, upper: Shape.Array,
              steps: Shape.Array) -> Self
    {
        get {
            let (extents, strides) = getExtents(lower, upper, steps)
            return view(at: lower, extents: extents, strides: strides)
        }
        set {
            expandSelfIfRepeated()
            let (extents, strides) = getExtents(lower, upper, steps)
            var view = sharedView(at: lower, extents: extents, strides: strides)
            Platform.service.copy(from: newValue, to: &view)
        }
    }
}

//==============================================================================
/// Derivative registration
extension TensorView where Self: DifferentiableTensorView {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    @inlinable
    @derivative(of: subscript)
    func _vjpSubscript(lower: Shape.Array, upper: Shape.Array, steps: Shape.Array)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper, steps], { v in
            var result = self.filled(with: Element.zero)
            result[lower, upper, steps] = v
            return result
        })
    }
}

