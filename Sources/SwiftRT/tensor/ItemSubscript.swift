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

public extension Tensor where Shape == Shape1, Element: Equatable {
    /// - Returns: an array of `Element`s
    @inlinable var array: [Element] {
        [Element](self)
    }

    @inlinable static func == (lhs: Self, rhs: [Element]) -> Bool {
        lhs.array == rhs
    }
}

//==============================================================================
// Rank2 array property and subscripts
public extension Tensor where Shape == Shape2 {
    /// - Returns: an array of `Element`s
    @inlinable var array: [[Element]] {
        var array2 = [[Element]]()
        for row in 0..<shape[0] {
            array2.append([Element](self[row, ...]))
        }
        return array2
    }

    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }

    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }
}

public extension Tensor where Shape == Shape2, Element: Equatable {
    @inlinable static func == (lhs: Self, rhs: [[Element]]) -> Bool {
        lhs.array == rhs
    }
}

//==============================================================================
// Rank3 array property and subscripts
public extension Tensor where Shape == Shape3 {
    //--------------------------------------------------------------------------
    /// return an array of elements
    @inlinable var array: [[[Element]]] {
        var array3 = [[[Element]]]()
        for depth in 0..<shape[0] {
            var array2 = [[Element]]()

            for row in 0..<shape[1] {
                let v = [Element](self[depth, row, ...])
                array2.append(v)
            }
            array3.append(array2)
        }
        return array3
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: PartialRangeExpression, D.Bound == Int {
        get { self[deps, 0..., 0...] }
        set { self[deps, 0..., 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int {
        get { self[deps, rows, 0...] }
        set { self[deps, rows, 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int {
        get { self[deps, 0..., cols] }
        set { self[deps, 0..., cols] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[0..., rows, 0...] }
        set { self[0..., rows, 0...] = newValue }
    }

    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., 0..., cols] }
        set { self[0..., 0..., cols] = newValue }
    }
}

public extension Tensor where Shape == Shape3, Element: Equatable {
    @inlinable static func == (lhs: Self, rhs: [[[Element]]]) -> Bool {
        lhs.array == rhs
    }
}

//==============================================================================
// These subscripts do a mutli-dimensional selection based on item indexes
// from dimension 0
public extension TensorType {
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript(range: UnboundedRange) -> Self { self }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get {
            let (start, end, _) = getItemRange(range.relativeTo(0..<shape[0]))
            return self[start, end]
        }
    }
    
    @usableFromInline
    @_semantics("autodiff.nonvarying")
    internal func getItemRange(_ range: StridedRange<Int>) ->
        (Shape, Shape, Shape)
    {
        var start = Shape.zero
        var end = self.shape
        var steps = Shape.one
        start[0] = range.start
        end[0] = range.end
        steps[0] = range.step
        return (start, end, steps)
    }
}

public extension MutableTensorType {
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get {
            let (start, end, _) = getItemRange(range.relativeTo(0..<shape[0]))
            return self[start, end]
        }
        set {
            let (start, end, _) = getItemRange(range.relativeTo(0..<shape[0]))
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
