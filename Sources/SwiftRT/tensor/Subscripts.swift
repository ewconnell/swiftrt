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

// HACK!!
public func copy<T, U>(from src: T, to dest: inout U)
    where T: Tensor, U: MutableTensor, T.Element == U.Element
{
    zip(dest.indices, src).forEach { dest[$0] = $1 }
}

//==============================================================================
// Rank1 array property and subscripts
public extension Tensor where Shape == Shape1 {
    /// return an array of elements
    @inlinable var array: [Element] { [Element](self) }
    
    @inlinable subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
    {
        let r = range.relativeTo(0..<shape[0])
        return self[Shape(r.start), Shape(r.end)]
    }
}

//------------------------------------------------------------------------------

public extension MutableTensor where Shape == Shape1 {
    // simplified integer range
    @inlinable subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
        {
        get {
            let r = range.relativeTo(0..<shape[0])
            return self[Shape(r.start), Shape(r.end)]
        }
        set {
            let r = range.relativeTo(0..<shape[0])
            self[Shape(r.start), Shape(r.end)] = newValue
        }
    }
}

//==============================================================================
// Rank2 array property and subscripts
public extension Tensor where Shape == Shape2 {
    /// return an array of elements
    @inlinable var array: [[Element]] {
        var array2 = [[Element]]()
        for row in 0..<shape[0] {
            array2.append([Element](self[row, ...]))
        }
        return array2
    }
    
    //--------------------------------------------------------------------------
    // subscripts
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R, C>(rows: R, cols: C) -> Self where
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
    {
        let r = rows.relativeTo(0..<shape[0])
        let c = cols.relativeTo(0..<shape[1])
        let lower = Shape2(r.start, c.start)
        let upper = Shape2(r.end, c.end)
        return self[lower, upper]
    }
    
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int
    {
        self[rows, 0...]
    }
    
    
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int
    {
        self[0..., cols]
    }
}

//------------------------------------------------------------------------------

public extension MutableTensor where Shape == Shape2 {
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<R, C>(rows: R, cols: C) -> Self where
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
        {
        get {
            let r = rows.relativeTo(0..<shape[0])
            let c = cols.relativeTo(0..<shape[1])
            return self[Shape2(r.start, c.start), Shape2(r.end, c.end)]
        }
        
        set {
            let r = rows.relativeTo(0..<shape[0])
            let c = cols.relativeTo(0..<shape[1])
            self[Shape2(r.start, c.start), Shape2(r.end, c.end)] = newValue
        }
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
    
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<D, R, C>(deps: D, rows: R, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
    {
        let d = deps.relativeTo(0..<shape[0])
        let r = rows.relativeTo(0..<shape[1])
        let c = cols.relativeTo(0..<shape[2])
        return self[Shape3(d.start, r.start, c.start),
                    Shape3(d.end, r.end, c.end)]
    }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: PartialRangeExpression, D.Bound == Int
    {
        self[deps, 0..., 0...]
    }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int
    {
        self[deps, rows, 0...]
    }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
    {
        self[deps, 0..., cols]
    }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int
    {
        self[0..., rows, 0...]
    }
    
    @inlinable
    //    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int
    {
        self[0..., 0..., cols]
    }
}

//------------------------------------------------------------------------------

public extension MutableTensor where Shape == Shape3 {
    //    @differentiable(where Self: DifferentiableTensorView)
    @inlinable subscript<D, R, C>(deps: D, rows: R, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
        {
        get {
            let d = deps.relativeTo(0..<shape[0])
            let r = rows.relativeTo(0..<shape[1])
            let c = cols.relativeTo(0..<shape[2])
            return self[Shape3(d.start, r.start, c.start),
                        Shape3(d.end, r.end, c.end)]
        }
        
        set {
            let d = deps.relativeTo(0..<shape[0])
            let r = rows.relativeTo(0..<shape[1])
            let c = cols.relativeTo(0..<shape[2])
            self[Shape3(d.start, r.start, c.start),
                 Shape3(d.end, r.end, c.end)] = newValue
        }
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

//==============================================================================
// These subscripts do a mutli-dimensional selection based on item indexes
// from dimension 0
public extension Tensor {
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

public extension MutableTensor {
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
