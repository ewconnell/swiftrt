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
            view(from: makePositive(index: Bounds(index)),
                 to: Bounds.one, with: Bounds.one).element
        }
        set {
            expandSelfIfRepeated()
            var view = sharedView(from: makePositive(index: Bounds(index)),
                                  to: Bounds.one, with: Bounds.one)
            view.element = newValue
        }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
        {
        get {
            let r = range.relativeTo(0..<bounds[0])
            return self[Bounds(r.start), Bounds(r.end), Bounds(r.step)]
        }
        set {
            let r = range.relativeTo(0..<bounds[0])
            self[Bounds(r.start), Bounds(r.end), Bounds(r.step)] = newValue
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
                getItemRange(range.relativeTo(0..<bounds[0]))
            return self[start, end, steps]
        }
        set {
            let (start, end, steps) =
                getItemRange(range.relativeTo(0..<bounds[0]))
            self[start, end, steps] = newValue
        }
    }
    
    @usableFromInline
    @_semantics("autodiff.nonvarying")
    internal func getItemRange(_ range: StridedRange<Int>) ->
        (Bounds, Bounds, Bounds)
    {
        var start = Bounds.zero
        var end = self.bounds
        var steps = Bounds.one
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
    /// `getUpper(_:_:_`
    /// computes the upper bound and strides from the specified bounds and steps
    /// - Parameter lower: the lower bound of the subview
    /// - Parameter upper: the upper bound of the subview
    /// - Parameter steps: the step interval along each dimension. This
    ///                    value can be negative to perform reverse traversal
    /// - Returns: the bounds and strides to be used to create a subview
    @inlinable
    func getUpper(_ lower: Bounds, _ upper: Bounds, _ steps: Bounds) ->
        (bounds: Bounds, strides: Bounds)
    {
        // verify bounds
        let bounds = upper &- lower
        assert(bounds.min() > 0, _messageInvalidBounds)

        // if all the steps are 1, then just reuse the parent strides
        if steps == Bounds.one {
            return (upper, self.strides)

        } else {
            // if one or more steps are not 1,
            // then recompute the subview bounds and strides

            // TODO: find out how to do SIMD abs(), doesn't seem to exist
            var absSteps = steps
            for i in 0..<Bounds.rank { absSteps[i] = Swift.abs(absSteps[i]) }

            let viewUpper = ((bounds &- 1 &+ absSteps) / absSteps) &+ lower
            let viewStrides = strides &* steps
            return (viewUpper, viewStrides)
        }
    }
    
    //--------------------------------------------------------------------------
    // views will have the same shared state as the parent
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(lower: Bounds, upper: Bounds, steps: Bounds) -> Self
    {
        get {
            let (viewUpper, strides) = getUpper(lower, upper, steps)
            return view(from: lower, to: viewUpper, with: strides)
        }
        set {
            expandSelfIfRepeated()
            let (viewUpper, strides) = getUpper(lower, upper, steps)
            var view = sharedView(from: lower, to: viewUpper, with: strides)
            Context.platform.copy(from: newValue, to: &view)
        }
    }
}

//==============================================================================
/// Derivative registration
extension TensorView where Self: DifferentiableTensorView {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    @inlinable
    @derivative(of: subscript)
    func _vjpSubscript(lower: Bounds, upper: Bounds, steps: Bounds)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper, steps], { v in
            var result = self.filled(with: Element.zero)
            result[lower, upper, steps] = v
            return result
        })
    }
}

