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

/// `Tensor Subscript Behavior`
/// A tensor subscripted with a range returns a sub view.
///
/// A tensor subscripted using `tensor.indices` or an Index formed
/// via the `ElementIndex` structure, will return an `Element`
///
/// A tensor subscripted with integers for each dimension is a convenience
/// function for wrapping the values in an `ElementIndex` structure, and
/// then returning the corresponding tensor `Element` value
//
//==============================================================================
// Rank2 array property and subscripts
public extension Tensor where Shape == Shape2 {

    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: SignedRangeExpression {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }

    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: SignedRangeExpression {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }
}

//==============================================================================
// Rank3 array property and subscripts
public extension Tensor where Shape == Shape3 {

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: SignedRangeExpression {
        get { self[deps, 0..., 0...] }
        set { self[deps, 0..., 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: SignedRangeExpression,
        R: SignedRangeExpression {
        get { self[deps, rows, 0...] }
        set { self[deps, rows, 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: SignedRangeExpression,
        C: SignedRangeExpression {
        get { self[deps, 0..., cols] }
        set { self[deps, 0..., cols] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: SignedRangeExpression {
        get { self[0..., rows, 0...] }
        set { self[0..., rows, 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: SignedRangeExpression {
        get { self[0..., 0..., cols] }
        set { self[0..., 0..., cols] = newValue }
    }
}

//==============================================================================
// These subscripts do a mutli-dimensional selection based on item indexes
// from dimension 0
public extension TensorType
{
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript(range: UnboundedRange) -> Self { self }
    
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R>(range: R) -> Self where R: SignedRangeExpression {
        get {
            let (start, end) = getItemRange(range.relativeTo(0..<shape[0]))
            return self[start, end]
        }
    }
    
    @_semantics("autodiff.nonvarying")
    @inlinable func getItemRange(_ range: Range<Int>) -> (Shape, Shape) {
        var lower = Shape.zero
        var upper = self.shape
        lower[0] = range.lowerBound
        upper[0] = range.upperBound
        return (lower, upper)
    }
}

public extension MutableTensorType
{
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R>(range: R) -> Self where R: SignedRangeExpression {
        get {
            let (start, end) = getItemRange(range.relativeTo(0..<shape[0]))
            return self[start, end]
        }
        set {
            let (start, end) = getItemRange(range.relativeTo(0..<shape[0]))
            self[start, end] = newValue
        }
    }
}

////==============================================================================
///// Derivative registration
//extension TensorType where Self: Differentiable {
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
