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
import Numerics

//==============================================================================
// DeviceQueue default implementations
// TODO: investigate use of SIMD for cpu_mapOps
public extension DeviceFunctions where Self: DeviceQueue {
    @inlinable
    func cpu_abs<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { Swift.abs($0) }
    }
    
    @inlinable
    func cpu_add<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: AdditiveArithmetic,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result, +)
    }
    
    @inlinable
    func cpu_and<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element == Bool,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result) { $0 && $1 }
    }
    
    @inlinable
    func cpu_cast<T, R>(from buffer: T, to result: inout R) where
        T: Collection, T.Element: AnyConvertable,
        R: MutableCollection, R.Element: AnyConvertable
    {
        cpu_mapOp(buffer, &result) { R.Element(any: $0) }
    }

    @inlinable
    func cpu_copy<T, R>(from x: T, to result: inout R) where
        T: Collection,
        R: MutableCollection, R.Element == T.Element
    {
        zip(result.indices, x).forEach { result[$0] = $1 }
    }
    
    @inlinable
    func cpu_delay(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }

    @inlinable
    func cpu_div<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: AlgebraicField,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result, /)
    }

    @inlinable
    func cpu_elementsAlmostEqual<T, R>(_ lhs: T, _ rhs: T,
                                       _ tolerance: T.Element,
                                       _ result: inout R) where
        T: Collection, T.Element: SignedNumeric & Comparable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
    }

    @inlinable
    func cpu_equal<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Equatable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, ==)
    }

    @inlinable
    func cpu_exp<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { .exp($0) }
    }

    @inlinable
    func cpu_fill<Element, R>(_ result: inout R, with element: Element) where
        R: MutableCollection, R.Element == Element
    {
        cpu_inPlaceOp(&result) { _ in element }
    }

    @inlinable
    func cpu_fill<T, R>(_ result: inout R, with range: T) where
        T: StridedRangeExpression & Collection,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(range, &result) { $0 }
    }

    @inlinable
    func cpu_greater<T, R>(_ lhs: T, _ rhs: T, _ result: inout R)
        where T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, >)
    }

    @inlinable
    func cpu_greaterOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, >=)
    }

    @inlinable
    func cpu_less<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, <)
    }

    @inlinable
    func cpu_lessOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, <=)
    }

    @inlinable
    func cpu_log<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { .log($0) }
    }

    @inlinable
    func cpu_max<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
    }

    @inlinable
    func cpu_min<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Comparable,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
    }

    @inlinable
    func cpu_mul<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Numeric,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result, *)
    }

    @inlinable
    func cpu_neg<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: SignedNumeric,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result, -)
    }

    @inlinable
    func cpu_notEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: Equatable,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result, !=)
    }

    @inlinable
    func cpu_or<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element == Bool,
        R: MutableCollection, R.Element == Bool
    {
        cpu_mapOp(lhs, rhs, &result) { $0 || $1 }
    }

    @inlinable
    func cpu_pow<T, R>(_ x: T, _ y: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, y, &result) { .pow($0, $1) }
    }

    @inlinable
    func cpu_replace<T, C, R>(_ x: T, _ y: T, _ condition: C,
                              _ result: inout R) where
        T: Collection,
        C: Collection, C.Element == Bool,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
    }

    @inlinable
    func cpu_sign<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { $0 < 0 ? -1 : 1 }
    }

    @inlinable
    func cpu_subtract<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: Collection, T.Element: AdditiveArithmetic,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(lhs, rhs, &result, -)
    }

    @inlinable
    func cpu_sqrt<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Real,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { .sqrt($0) }
    }

    @inlinable
    func cpu_squared<T, R>(_ x: T, _ result: inout R) where
        T: Collection, T.Element: Numeric,
        R: MutableCollection, R.Element == T.Element
    {
        cpu_mapOp(x, &result) { $0 * $0 }
    }

    @inlinable
    func cpu_reduce<T, R>(_ x: T,
                          _ result: inout R,
                          _ opId: ReductionOp,
                          _ opNext: @escaping (T.Element, T.Element) -> T.Element,
                          _ opFinal: ReduceOpFinal<R>?) where
        T: ShapedBuffer, T.Shape == R.Shape,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        // created a repeated shape for the initial results to match `x`
        let repeatedShape = result.shape.repeated(to: x.shape.extents)
        
        // create a new elements collection to iterate using the `result`
        // buffer and the new repeated shape.
        var repeatedBuffer = MutableBufferElements(repeatedShape, result.pointer)

        // do the reductions
        cpu_reductionOp(x, &repeatedBuffer, opNext)

        if let op = opFinal {
            cpu_inPlaceOp(&result, op)
        }
    }
}

//==============================================================================
// DeviceQueue default derivative implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// vjpMinMax
    @inlinable
    func cpu_vjpMinMax<T, R>(
        _ x: T, _ y: T, _ scale: T,
        _ op: @escaping (T.Element, T.Element) -> Bool,
        _ resultTrue: inout R, _ resultFalse: inout R)
        where
        T : Collection, T.Element : Comparable & Numeric,
        R : MutableCollection, R.Element == T.Element
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
