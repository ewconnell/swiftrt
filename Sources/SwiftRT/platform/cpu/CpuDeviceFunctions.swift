//******************************************************************************
// Copyright 2020 Google LLC
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
import Real

//==============================================================================
// DeviceQueue default implementations
public extension DeviceFunctions where Self: DeviceQueue {
    func cpu_abs<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { Swift.abs($0) }
    }
    
    func cpu_add<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result, +)
    }
    
    func cpu_and<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result) { $0 && $1 }
    }
    
    func cpu_cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: AnyConvertable,
        R: MutableShapedBuffer, R.Element: AnyConvertable
    {
        cpu_mapOp(buffer, &result) { R.Element(any: $0) }
    }
//
//    func cpu_concat<T, R>(buffers: [T], alongAxis axis: Int, result: inout R) where
//        T: ShapedBuffer,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        //        var index = T.Shape.zeros
//        // rewrite
//        fatalError()
//        //        for buffer in buffers {
//        //            var view = result.mutableView(at: index, extents: tensor.extents)
//        //            tensor.map(into: &view) { $0 }
//        //            index[axis] += tensor.extents[axis]
//        //        }
//    }
//
//    func cpu_delay(atLeast interval: TimeInterval) {
//        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
//        Thread.sleep(forTimeInterval: interval)
//    }
//
//    func cpu_div<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: AlgebraicField,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(lhs, rhs, &result, /)
//    }
//
//    func cpu_elementsAlmostEqual<T, R>(lhs: T, rhs: T, tolerance: T.Element,
//                                       result: inout R) where
//        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
//    }
//
//    func cpu_equal<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, ==)
//    }
//
//    func cpu_exp<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Real,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result) { .exp($0) }
//    }
//
//    func cpu_fill<Element, R>(result: inout R, with element: Element) where
//        R: MutableShapedBuffer, R.Element == Element
//    {
//        cpu_inPlaceOp(&result) { _ in element }
//    }
//
//    func cpu_fill<T, R>(result: inout R, with range: T) where
//T: StridedRangeExpression,
//R: MutableShapedBuffer, R.Element == T.Bound
//    {
//        // add a new mapOp for ranges
//        fatalError()
//        //        cpu_mapOp(&result) { $0 }
//    }
//
//    func cpu_greater<T, R>(lhs: T, rhs: T, result: inout R)
//        where T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, >)
//    }
//
//    func cpu_greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, >=)
//    }
//
//    func cpu_less<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, <)
//    }
//
//    func cpu_lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, <=)
//    }
//
//    func cpu_log<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Real,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result) { .log($0) }
//    }
//
//    func cpu_max<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
//    }
//
//    func cpu_min<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Comparable,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
//    }
//
//    func cpu_mul<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Numeric,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(lhs, rhs, &result, *)
//    }
//
//    func cpu_neg<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: SignedNumeric,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result, -)
//    }
//
//    func cpu_notEqual<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result, !=)
//    }
//
//    func cpu_or<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element == Bool,
//        R: MutableShapedBuffer, R.Element == Bool
//    {
//        cpu_mapOp(lhs, rhs, &result) { $0 || $1 }
//    }
//
//    func cpu_pow<T, R>(x: T, y: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Real,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, y, &result) { .pow($0, $1) }
//    }
//
//    func cpu_replace<T, C, R>(x: T, with y: T, where condition: C,
//                              result: inout R) where
//        T: ShapedBuffer,
//        C: ShapedBuffer, C.Element == Bool,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
//    }
//
//    func cpu_sign<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Real,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result) { $0 < 0 ? -1 : 1 }
//    }
//
//    func cpu_subtract<T, R>(lhs: T, rhs: T, result: inout R) where
//        T: ShapedBuffer, T.Element: AdditiveArithmetic,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(lhs, rhs, &result, -)
//    }
//
//    func cpu_sqrt<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Real,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result) { .sqrt($0) }
//    }
//
//    func cpu_squared<T, R>(x: T, result: inout R) where
//        T: ShapedBuffer, T.Element: Numeric,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        cpu_mapOp(x, &result) { $0 * $0 }
//    }
//
//    func cpu_reduce<T, R>(x: T,
//                          into result: inout R,
//                          opId: ReductionOp,
//                          opNext: @escaping (T.Element, T.Element) -> T.Element,
//                          opFinal: ReduceOpFinal<T>?) where
//        T: ShapedBuffer,
//        R: MutableShapedBuffer, R.Element == T.Element
//    {
//        fatalError()
////        assert(result.isContiguous, "Result storage must be contiguous")
////
////        // created a repeated view of the initial results to match `x`
////        var repeated = T(shape: result.shape.repeated(to: x.extents),
////                         tensorArray: result.tensorArray,
////                         viewOffset: result.viewOffset,
////                         isMutable: true)
////
////        // get the elements collection and do the reduction
////        var repeatedElements = repeated.mutableElements(using: self)
////        reductionOp(x.elements, &repeatedElements, opNext)
////
////        if let op = opFinal {
////            var elements = result.mutableElements(using: self)
////            inPlaceOp(&elements, op)
////        }
//    }
}

//==============================================================================
// DeviceQueue default derivative implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// vjpMinMax
    @inlinable
    
    func cpu_vjpMinMax<T, R>(
        x: T, y: T, scale: T,
        op: @escaping (T.Element, T.Element) -> Bool,
        resultTrue: inout R, resultFalse: inout R)
        where
        T : ShapedBuffer, T.Element : Comparable & Numeric,
        R : MutableShapedBuffer, R.Element == T.Element
    {
        cpu_mapOp(x, y, scale, &resultTrue, &resultFalse) {
            op($0, $1) ? ($2, T.Element.zero) : (T.Element.zero, $2)
        }
    }
}


//==============================================================================
// DeviceFunctions mapOps
public extension DeviceFunctions {
    
    // inPlaceOp
    @inlinable
    func cpu_inPlaceOp<R>(_ r: inout R,_ op: @escaping (R.Element) -> R.Element)
        where R: MutableCollection
    {
        r.indices.forEach { r[$0] = op(r[$0]) }
    }
    
    // mapOp 1
    @inlinable
    func cpu_mapOp<T, R>(_ x: T, _ r: inout R,
                         _ op: @escaping (T.Element) -> R.Element) where
        T: Collection, R: MutableCollection
    {
        zip(r.indices, x).forEach { r[$0] = op($1) }
    }
    
    // mapOp 2
    @inlinable
    func cpu_mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ r: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: Collection, RHS: Collection, R: MutableCollection
    {
        zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = op($1.0, $1.1) }
    }
    
    // mapOp 3
    @inlinable
    func cpu_mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ r: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element) where
        T1: Collection, T2: Collection, T3: Collection, R: MutableCollection
    {
        zip(r.indices, zip(a, zip(b, c))).forEach { r[$0] = op($1.0, $1.1.0, $1.1.1) }
    }
    
    // mapOp 3R2
    /// generically combines three tensors
    @inlinable
    func cpu_mapOp<T1, T2, T3, R1, R2>(
        _ a: T1, _ b: T2, _ c: T3, _ r1: inout R1,  _ r2: inout R2,
        _ op: @escaping
        (T1.Element, T2.Element, T3.Element) -> (R1.Element, R2.Element))
        where
        T1: Collection, T2: Collection, T3: Collection,
        R1: MutableCollection, R2: MutableCollection
    {
        zip(zip(r1.indices, r2.indices), zip(a, zip(b, c))).forEach {
            let (r1v, r2v) = op($1.0, $1.1.0, $1.1.1)
            r1[$0.0] = r1v
            r2[$0.1] = r2v
        }
    }
    
    // reductionOp
    @inlinable
    func cpu_reductionOp<T, R>(
        _ x: T, _ r: inout R,
        _ op: @escaping (R.Element, T.Element) -> R.Element)
        where T: Collection, R: MutableCollection
    {
        zip(r.indices, x).forEach { r[$0] = op(r[$0], $1) }
    }
}
