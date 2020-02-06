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
import Real

// TODO move these
infix operator **  : MultiplicationPrecedence
infix operator .<  : ComparisonPrecedence
infix operator .<= : ComparisonPrecedence
infix operator .>= : ComparisonPrecedence
infix operator .>  : ComparisonPrecedence
infix operator .== : ComparisonPrecedence
infix operator .!= : ComparisonPrecedence

//==============================================================================
// DeviceFunctions
public protocol DeviceFunctions {
    //--------------------------------------------------------------------------
    // generic helpers
    /// mapOp 1
    /// generically maps tensor elements
    func mapOp<T, R>(_ x: T, _ result: inout R,
                     _ op: @escaping (T.Element) -> R.Element) where
        T: TensorView, R: TensorView
    /// mapOp 2
    /// generically combines two tensors
    func mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ result: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: TensorView, RHS: TensorView, R: TensorView
    /// mapOp 3
    /// generically combines three tensors
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element)
        where T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    /// mapOp 3R2
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result1: inout R,  _ result2: inout R,
        _ op: @escaping
        (T1.Element, T2.Element, T3.Element) -> (R.Element, R.Element))
        where T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    /// inPlaceOp
    /// does in place op on a mutable collection
    func inPlaceOp<T>(_ result: inout T,
                      _ op: @escaping (T.Element) -> T.Element) where
        T: MutableCollection
    /// reductionOp
    /// does a tensor reduction op
    func reductionOp<T, R>(
        _ x: T, _ result: inout R,
        _ op: @escaping (R.Element, T.Element) -> R.Element) where
        T: Collection, R: MutableCollection
    
    //--------------------------------------------------------------------------
    // ops
    /// Computes the absolute value of the specified TensorView element-wise.
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// add
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: AdditiveArithmetic
    /// and
    func and<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element == Bool
    /// cast
    func cast<T, U>(from view: T, to result: inout U) where
        T: TensorView, T.Element: AnyConvertable,
        U: TensorView, U.Element: AnyConvertable
    /// concat
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T) where
        T: TensorView
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: AlgebraicField
    /// elementsAlmostEqual
    func elementsAlmostEqual<T>(lhs: T, rhs: T, tolerance: T.Element,
                                result: inout T.BoolView) where
        T: TensorView, T.Element: SignedNumeric & Comparable
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where T: TensorView
    /// exp
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// fill(result:with element:
    func fill<T>(result: inout T, with element: T.Element) where T: TensorView
    /// fill(result:with range:
    func fill<T, R>(result: inout T, with range: R) where
        T: TensorView,
        R: StridedRangeExpression, R.Bound == T.Element
    /// greater
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// greaterOrEqual
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// less
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// lessOrEqual
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// log
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// Computes the element-wise maximum of two tensors.
    func max<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// Computes the element-wise minimum of two tensors.
    func min<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// mul
    func mul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView
    /// or
    func or<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element == Bool
    /// pow
    func pow<T>(x: T, y: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// replace
    func replace<T>(x: T, with y: T, where condition: T.BoolView,
                    result: inout T) where T: TensorView
    /// sign
    func sign<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: AdditiveArithmetic
    /// sqrt
    func sqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real
    /// squared
    func squared<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    /// reduce
    /// Reduces `x` along the specified axes
    /// - Parameter x: value tensor
    /// - Parameter result: contains the initial value of the result on entry
    ///  and then final reduction result on return
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func reduce<T>(x: T,
                   into result: inout T,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?)
        where T: TensorView
    
    //==========================================================================
    // derivative function declarations
    
    /// vjpMinMax
    func vjpMinMax<T>(
        x: T, y: T, scale: T, op: @escaping (T.Element, T.Element) -> Bool,
        resultTrue: inout T, resultFalse: inout T)
        where T: TensorView, T.Element: Comparable & Numeric
}

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionOp
public enum ReductionOp: Int, Codable {
    case add
    case mean
    case mul
    case min
    case max
    case amax
    case asum
    case sqrtSumSquares
    case mulNonZeros
    case compare
}

public typealias ReduceOpFinal<T: TensorView> = (T.Element) -> T.Element

//==============================================================================
// DeviceQueue default delegating implementations
public extension DeviceFunctions where Self: DeviceQueue {
    // mapOp 1
    /// generically maps a tensor
    func mapOp<T, R>(_ x: T, _ result: inout R,
                     _ op: @escaping (T.Element) -> R.Element) where
        T: TensorView, R: TensorView
    {
        cpu_mapOp(x, &result, op)
    }
    // mapOp 2
    /// generically combines two tensors
    func mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ result: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: TensorView, RHS: TensorView, R: TensorView
    {
        cpu_mapOp(lhs, rhs, &result, op)
    }
    // mapOp 3
    /// generically combines three tensors
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element) where
        T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    {
        cpu_mapOp(a, b, c, &result, op)
    }
    // mapOp 3R2
    /// generically combines three tensors
    
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result1: inout R,  _ result2: inout R,
        _ op: @escaping
        (T1.Element, T2.Element, T3.Element) -> (R.Element, R.Element))
        where T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    {
        cpu_mapOp(a, b, c, &result1, &result2, op)
    }
    // inPlaceOp
    func inPlaceOp<T>(_ result: inout T,
                      _ op: @escaping (T.Element) -> T.Element) where
        T: MutableCollection
    {
        cpu_inPlaceOp(&result, op)
    }
    // reductionOp
    func reductionOp<T, R>(
        _ x: T, _ result: inout R,
        _ op: @escaping (R.Element, T.Element) -> R.Element) where
        T: Collection, R: MutableCollection
    {
        cpu_reductionOp(x, &result, op)
    }
    
    //==========================================================================
    /// abs
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Element: Real {
            cpu_abs(x: x, result: &result)
    }
    // add
    func add<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: AdditiveArithmetic {
            cpu_add(lhs: lhs, rhs: rhs, result: &result)
    }
    /// cast
    func cast<T, U>(from view: T, to result: inout U) where
        T: TensorView, T.Element: AnyConvertable,
        U: TensorView, U.Element: AnyConvertable {
            cpu_cast(from: view, to: &result)
    }
    // concat
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T)
        where T: TensorView {
            cpu_concat(tensors: tensors, alongAxis: axis, result: &result)
    }
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: AlgebraicField {
            cpu_div(lhs: lhs, rhs: rhs, result: &result)
    }
    /// elementsAlmostEqual
    func elementsAlmostEqual<T>(lhs: T, rhs: T, tolerance: T.Element,
                                result: inout T.BoolView)
        where T: TensorView, T.Element: SignedNumeric & Comparable {
            cpu_elementsAlmostEqual(lhs: lhs, rhs: rhs, tolerance: tolerance,
                                    result: &result)
    }
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView {
            cpu_equal(lhs: lhs, rhs: rhs, result: &result)
    }
    /// exp
    func exp<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Real {
            cpu_exp(x: x, result: &result)
    }
    /// fill(result:with:
    func fill<T>(result: inout T, with element: T.Element) where T: TensorView {
        cpu_fill(result: &result, with: element)
    }
    /// fill(result:with range:
    func fill<T, R>(result: inout T, with range: R) where
        T: TensorView,
        R: StridedRangeExpression, R.Bound == T.Element {
            cpu_fill(result: &result, with: range)
    }
    /// less
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable {
            cpu_less(lhs: lhs, rhs: rhs, result: &result)
    }
    /// lessOrEqual
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable {
            cpu_lessOrEqual(lhs: lhs, rhs: rhs, result: &result)
    }
    /// greater
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable {
            cpu_greater(lhs: lhs, rhs: rhs, result: &result)
    }
    /// greaterOrEqual
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable {
            cpu_greaterOrEqual(lhs: lhs, rhs: rhs, result: &result)
    }
    /// log
    func log<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Real {
            cpu_log(x: x, result: &result)
    }
    /// Computes the element-wise maximum of two tensors.
    func max<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Comparable {
            cpu_max(lhs: lhs, rhs: rhs, result: &result)
    }
    /// Computes the element-wise minimum of two tensors.
    func min<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Comparable {
            cpu_min(lhs: lhs, rhs: rhs, result: &result)
    }
    /// mul
    func mul<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric {
            cpu_mul(lhs: lhs, rhs: rhs, result: &result)
    }
    /// neg
    func neg<T>(x: T, result: inout T)
        where T: TensorView, T.Element: SignedNumeric {
            cpu_neg(x: x, result: &result)
    }
    /// notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView {
            cpu_notEqual(lhs: lhs, rhs: rhs, result: &result)
    }
    /// pow
    func pow<T>(x: T, y: T, result: inout T)
        where T: TensorView, T.Element: Real {
            cpu_pow(x: x, y: y, result: &result)
    }
    /// replace
    func replace<T>(x: T, with y: T, where condition: T.BoolView,
                    result: inout T) where T: TensorView {
        cpu_replace(x: x, with: y, where: condition, result: &result)
    }
    /// sign
    func sign<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Real {
            cpu_sign(x: x, result: &result)
    }
    /// subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: AdditiveArithmetic {
            cpu_subtract(lhs: lhs, rhs: rhs, result: &result)
    }
    /// sqrt
    func sqrt<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Real {
            cpu_sqrt(x: x, result: &result)
    }
    /// squared
    func squared<T>(x: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    {
        cpu_squared(x: x, result: &result)
    }
    /// reduce
    func reduce<T>(x: T,
                   into result: inout T,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?)
        where T: TensorView
    {
        cpu_reduce(x: x, into: &result, opId: opId,
                   opNext: opNext, opFinal: opFinal)
    }
}

//==============================================================================
// DeviceQueue default derivative implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// vjpMinMax
    func vjpMinMax<T>(
        x: T, y: T, scale: T, op: @escaping (T.Element, T.Element) -> Bool,
        resultTrue: inout T, resultFalse: inout T)
        where T: TensorView, T.Element: Comparable & Numeric
    {
        cpu_vjpMinMax(x: x, y: y, scale: scale, op: op,
                      resultTrue: &resultTrue, resultFalse: &resultFalse)
    }
}
