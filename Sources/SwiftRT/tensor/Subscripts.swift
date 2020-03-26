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

//public extension Tensor {
//    @inlinable
////    @differentiable(where Self: DifferentiableTensorView)
//    subscript(range: UnboundedRange) -> Self { self }
//
//    @inlinable
////    @differentiable(where Self: DifferentiableTensorView)
//    subscript<R>(range: R) -> Self
//        where R: PartialRangeExpression, R.Bound == Int {
//        get {
//            let (start, end, steps) =
//                getItemRange(range.relativeTo(0..<shape[0]))
//            return self[start, end, steps]
//        }
//        set {
//            let (start, end, steps) =
//                getItemRange(range.relativeTo(0..<shape[0]))
//            self[start, end, steps] = newValue
//        }
//    }
//
//    @usableFromInline
//    @_semantics("autodiff.nonvarying")
//    internal func getItemRange(_ range: StridedRange<Int>) ->
//        (Shape, Shape, Shape)
//    {
//        var start = Shape.zero
//        var end = self.shape
//        var steps = Shape.one
//        start[0] = range.start
//        end[0] = range.end
//        steps[0] = range.step
//        return (start, end, steps)
//    }
//}
//
////==============================================================================
//public extension Tensor {
//    //--------------------------------------------------------------------------
//    /// `getUpper(_:_:_`
//    /// computes the upper bound and strides from the specified shape and steps
//    /// - Parameter lower: the lower bound of the subview
//    /// - Parameter upper: the upper bound of the subview
//    /// - Parameter steps: the step interval along each dimension. This
//    ///                    value can be negative to perform reverse traversal
//    /// - Returns: the shape and strides to be used to create a subview
//    @inlinable
//    func getUpper(_ lower: Shape, _ upper: Shape, _ steps: Shape) ->
//        (shape: Shape, strides: Shape)
//    {
//        // verify shape
//        let shape = upper &- lower
//        assert(shape.min() > 0, _messageInvalidShape)
//
//        // if all the steps are 1, then just reuse the parent strides
//        if steps == Shape.one {
//            return (upper, self.strides)
//
//        } else {
//            // if one or more steps are not 1,
//            // then recompute the subview shape and strides
//
//            // TODO: find out how to do SIMD abs(), doesn't seem to exist
//            var absSteps = steps
//            for i in 0..<Shape.rank { absSteps[i] = Swift.abs(absSteps[i]) }
//
//            let viewUpper = ((shape &- 1 &+ absSteps) / absSteps) &+ lower
//            let viewStrides = strides &* steps
//            return (viewUpper, viewStrides)
//        }
//    }
//
//    //--------------------------------------------------------------------------
//    // views will have the same shared state as the parent
//    @inlinable
//    @differentiable(where Self: DifferentiableTensorView)
//    subscript(lower: Shape, upper: Shape, steps: Shape) -> Self
//    {
//        get {
//            let (viewUpper, strides) = getUpper(lower, upper, steps)
//            return view(from: lower, to: viewUpper, with: strides)
//        }
//        set {
//            expandSelfIfRepeated()
//            let (viewUpper, strides) = getUpper(lower, upper, steps)
//            var view = sharedView(from: lower, to: viewUpper, with: strides)
//            Context.platform.copy(from: newValue, to: &view)
//        }
//    }
//}
//
////==============================================================================
///// Derivative registration
//extension Tensor where Self: Differentiable {
//    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
//    @inlinable
//    @derivative(of: subscript)
//    func _vjpSubscript(lower: Shape, upper: Shape, steps: Shape)
//        -> (value: Self, pullback: (Self) -> Self)
//    {
//        return (self[lower, upper, steps], { v in
//            var result = self.filled(with: Element.zero)
//            result[lower, upper, steps] = v
//            return result
//        })
//    }
//}
//
