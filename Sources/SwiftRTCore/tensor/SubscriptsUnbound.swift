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
// These subscripts do a mutli-dimensional selection based on item indexes
// from dimension 0
public extension Tensor {
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable subscript(range: UnboundedRange) -> Self { self }
    
//    @differentiable(where Self: DifferentiableTensor)
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

    @_semantics("autodiff.nonvarying")
    @inlinable func getItemRange(_ range: Range<Int>) -> (Shape, Shape) {
        var lower = Shape.zero
        var upper = self.shape
        lower[0] = range.lowerBound
        upper[0] = range.upperBound
        return (lower, upper)
    }
}

//==============================================================================
// Rank2 array property and subscripts
public extension Tensor where Shape == Shape2 {

    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable subscript<R0>(r0: R0, r1: UnboundedRange) -> Self
        where R0: SignedRangeExpression {
        get { self[r0, 0...] }
        set { self[r0, 0...] = newValue }
    }

    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable subscript<R1>(r0: UnboundedRange, r1: R1) -> Self
        where R1: SignedRangeExpression {
        get { self[0..., r1] }
        set { self[0..., r1] = newValue }
    }
}

//==============================================================================
// Rank3 array property and subscripts
public extension Tensor where Shape == Shape3 {

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0>(r0: R0, r1: UnboundedRange, r2: UnboundedRange) -> Self
        where R0: SignedRangeExpression {
        get { self[r0, 0..., 0...] }
        set { self[r0, 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0, R1>(r0: R0, r1: R1, r2: UnboundedRange) -> Self
        where
        R0: SignedRangeExpression,
        R1: SignedRangeExpression {
        get { self[r0, r1, 0...] }
        set { self[r0, r1, 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0, R2>(r0: R0, r1: UnboundedRange, r2: R2) -> Self
        where
        R0: SignedRangeExpression,
        R2: SignedRangeExpression {
        get { self[r0, 0..., r2] }
        set { self[r0, 0..., r2] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R1>(r0: UnboundedRange, r1: R1, r2: UnboundedRange) -> Self
        where R1: SignedRangeExpression {
        get { self[0..., r1, 0...] }
        set { self[0..., r1, 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R2>(r0: UnboundedRange, r1: UnboundedRange, r2: R2) -> Self
        where R2: SignedRangeExpression {
        get { self[0..., 0..., r2] }
        set { self[0..., 0..., r2] = newValue }
    }
}

//==============================================================================
// Rank4 array property and subscripts
public extension Tensor where Shape == Shape4 {

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0>(r0: R0, r1: UnboundedRange, r2: UnboundedRange,
        r3: UnboundedRange) -> Self
        where R0: SignedRangeExpression {
        get { self[r0, 0..., 0..., 0...] }
        set { self[r0, 0..., 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0, R1>(r0: R0, r1: R1, r2: UnboundedRange,
        r3: UnboundedRange) -> Self where
        R0: SignedRangeExpression,
        R1: SignedRangeExpression {
        get { self[r0, r1, 0..., 0...] }
        set { self[r0, r1, 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0, R2>(r0: R0, r1: UnboundedRange, r2: R2,
        r3: UnboundedRange) -> Self where
        R0: SignedRangeExpression,
        R2: SignedRangeExpression {
        get { self[r0, 0..., r2, 0...] }
        set { self[r0, 0..., r2, 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R1>(r0: UnboundedRange, r1: R1, r2: UnboundedRange,
        r3: UnboundedRange) -> Self
        where R1: SignedRangeExpression {
        get { self[0..., r1, 0..., 0...] }
        set { self[0..., r1, 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R2>(r0: UnboundedRange, r1: UnboundedRange, r2: R2,
        r3: UnboundedRange) -> Self
        where R2: SignedRangeExpression {
        get { self[0..., 0..., r2, 0...] }
        set { self[0..., 0..., r2, 0...] = newValue }
    }
    
    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0,R3>(r0: R0, r1: UnboundedRange, r2: UnboundedRange, r3: R3)
        -> Self where
        R0: SignedRangeExpression,
        R3: SignedRangeExpression {
        get { self[r0, 0..., 0..., 0...] }
        set { self[r0, 0..., 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0,R1,R3>(r0: R0, r1: R1, r2: UnboundedRange, r3: R3) -> Self
        where
        R0: SignedRangeExpression,
        R1: SignedRangeExpression,
        R3: SignedRangeExpression {
        get { self[r0, r1, 0..., 0...] }
        set { self[r0, r1, 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R0,R2,R3>(r0: R0, r1: UnboundedRange, r2: R2, r3: R3) -> Self
        where
        R0: SignedRangeExpression,
        R2: SignedRangeExpression,
        R3: SignedRangeExpression {
        get { self[r0, 0..., r2, 0...] }
        set { self[r0, 0..., r2, 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R1,R3>(r0: UnboundedRange, r1: R1, r2: UnboundedRange,
        r3: R3) -> Self where
        R1: SignedRangeExpression,
        R3: SignedRangeExpression {
        get { self[0..., r1, 0..., 0...] }
        set { self[0..., r1, 0..., 0...] = newValue }
    }

    @inlinable
    @differentiable(where TensorElement.Value: DifferentiableElement)
    subscript<R2,R3>(r0: UnboundedRange, r1: UnboundedRange, r2: R2,
        r3: R3) -> Self where
        R2: SignedRangeExpression,
        R3: SignedRangeExpression {
        get { self[0..., 0..., r2, 0...] }
        set { self[0..., 0..., r2, 0...] = newValue }
    }
}

