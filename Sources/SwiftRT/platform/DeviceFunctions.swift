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
/// DeviceFunctions
/// Device functions require input arguments to conform to `ShapedBuffer`
/// and output arguments to conform to `MutableShapedBuffer`. They cannot
/// simply be Collections, because accelerator device kernels will require
/// the shaped extents and strides information to compute indices in parallel.
public protocol DeviceFunctions {
    /// the thread that created this queue. Used to detect accidental access
    var creatorThread: Thread { get }
    
    //--------------------------------------------------------------------------
    /// abs
    func abs<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element

    /// add
    func add<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element

    /// and
    func and<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool

    /// cast
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: AnyConvertable,
        R: MutableShapedBuffer, R.Element: AnyConvertable

    /// copy
    func copy<T, R>(from x: T, to result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element

    /// delay
    func delay(_ interval: TimeInterval)

    /// div
    func div<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AlgebraicField,
        R: MutableShapedBuffer, R.Element == T.Element

    /// elementsAlmostEqual
    func elementsAlmostEqual<T, R>(_ lhs: T, _ rhs: T, _ tolerance: T.Element,
                                   _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
        R: MutableShapedBuffer, R.Element == Bool

    /// equal
    func equal<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool

    /// exp
    func exp<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element

    /// fill(result:with element:
    func fill<Element, R>(_ result: inout R, with element: Element) where
        R: MutableShapedBuffer, R.Element == Element

    /// fill(result:with range:
    func fill<T, R>(_ result: inout R, with range: T) where
        T: StridedRangeExpression & Collection,
        R: MutableShapedBuffer, R.Element == T.Element

    /// greater
    func greater<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// greaterOrEqual
    func greaterOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// less
    func less<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// lessOrEqual
    func lessOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// log
    func log<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// Computes the element-wise maximum of two tensors.
    func max<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// Computes the element-wise minimum of two tensors.
    func min<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// mul
    func mul<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// notEqual
    func notEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// or
    func or<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// pow
    func pow<T, R>(_ x: T, _ y: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// replace
    func replace<T, C, R>(_ x: T, _ y: T, _ condition: C,
                          _ result: inout R) where
        T: ShapedBuffer,
        C: ShapedBuffer, C.Element == Bool,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// sign
    func sign<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// subtract
    func subtract<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// sqrt
    func sqrt<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// squared
    func squared<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// reduce
    /// Reduces `x` along the specified axes
    /// - Parameter x: value tensor
    /// - Parameter result: contains the initial value of the result on entry
    ///  and then final reduction result on return
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func reduce<T, R>(_ x: T,
                   _ result: inout R,
                   _ opId: ReductionOp,
                   _ opNext: @escaping (T.Element, T.Element) -> T.Element,
                   _ opFinal: ReduceOpFinal<T>?) where
        T: ShapedBuffer, T.Shape == R.Shape,
        R: MutableShapedBuffer, R.Element == T.Element
    
    //==========================================================================
    // derivative function declarations
    
    /// vjpMinMax
    func vjpMinMax<T, R>(
        _ x: T, _ y: T, _ scale: T,
        _ op: @escaping (T.Element, T.Element) -> Bool,
        _ resultTrue: inout R, _ resultFalse: inout R)
        where
        T: ShapedBuffer, T.Element: Comparable & Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
}


//==============================================================================
// DeviceQueue default cpu delegating implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// abs
    @inlinable
    func abs<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_abs(x, &result)
    }

    /// add
    @inlinable
    func add<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_add(lhs, rhs, &result)
    }

    /// and
    @inlinable
    func and<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_and(lhs, rhs, &result)
    }

    /// cast
    @inlinable
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: AnyConvertable,
        R: MutableShapedBuffer, R.Element: AnyConvertable
    {
        cpu_cast(from: buffer, to: &result)
    }

    /// copy
    @inlinable
    func copy<T, R>(from x: T, to result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_copy(from: x, to: &result)
    }
    
    /// delay
    @inlinable
    func delay(_ interval: TimeInterval)
    {
        cpu_delay(atLeast: interval)
    }

    /// div
    @inlinable
    func div<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AlgebraicField,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_div(lhs, rhs, &result)
    }

    /// elementsAlmostEqual
    @inlinable
    func elementsAlmostEqual<T, R>(_ lhs: T, _ rhs: T,
                                   _ tolerance: T.Element,
                                   _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_elementsAlmostEqual(lhs, rhs, tolerance, &result)
    }

    /// equal
    @inlinable
    func equal<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_equal(lhs, rhs, &result)
    }

    /// exp
    @inlinable
    func exp<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_exp(x, &result)
    }

    /// fill(result:with element:
    @inlinable
    func fill<Element, R>(_ result: inout R, with element: Element) where
        R: MutableShapedBuffer, R.Element == Element
    {
        cpu_fill(&result, with: element)
    }

    /// fill(result:with range:
    @inlinable
    func fill<T, R>(_ result: inout R, with range: T) where
        T: StridedRangeExpression & Collection,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_fill(&result, with: range)
    }

    /// greater
    @inlinable
    func greater<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_greater(lhs, rhs, &result)
    }

    /// greaterOrEqual
    @inlinable
    func greaterOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_greaterOrEqual(lhs, rhs, &result)
    }

    /// less
    @inlinable
    func less<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_less(lhs, rhs, &result)
    }

    /// lessOrEqual
    @inlinable
    func lessOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_lessOrEqual(lhs, rhs, &result)
    }

    /// log
    @inlinable
    func log<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_log(x, &result)
    }

    /// Computes the element-wise maximum of two tensors.
    @inlinable
    func max<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_max(lhs, rhs, &result)
    }

    /// Computes the element-wise minimum of two tensors.
    @inlinable
    func min<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_min(lhs, rhs, &result)
    }

    /// mul
    @inlinable
    func mul<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_mul(lhs, rhs, &result)
    }

    /// neg
    /// returns the element-wise negation of `x`
    @inlinable
    func neg<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_neg(x, &result)
    }

    /// notEqual
    @inlinable
    func notEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_notEqual(lhs, rhs, &result)
    }

    /// or
    @inlinable
    func or<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        cpu_or(lhs, rhs, &result)
    }

    /// pow
    @inlinable
    func pow<T, R>(_ x: T, _ y: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_pow(x, y, &result)
    }

    /// replace
    @inlinable
    func replace<T, C, R>(_ x: T, _ y: T, _ condition: C,
                          _ result: inout R) where
        T: ShapedBuffer,
        C: ShapedBuffer, C.Element == Bool,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_replace(x, y, condition, &result)
    }

    /// sign
    @inlinable
    func sign<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_sign(x, &result)
    }

    /// subtract
    @inlinable
    func subtract<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_subtract(lhs, rhs, &result)
    }

    /// sqrt
    @inlinable
    func sqrt<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_sqrt(x, &result)
    }

    /// squared
    @inlinable
    func squared<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_squared(x, &result)
    }

    /// reduce
    /// Reduces `x` along the specified axes
    /// - Parameter x: value tensor
    /// - Parameter result: contains the initial value of the result on entry
    ///  and then final reduction result on return
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable
    func reduce<T, R>(_ x: T,
                   _ result: inout R,
                   _ opId: ReductionOp,
                   _ opNext: @escaping (T.Element, T.Element) -> T.Element,
                   _ opFinal: ReduceOpFinal<R>?) where
        T: ShapedBuffer, T.Shape == R.Shape,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_reduce(x, &result, opId, opNext, opFinal)
    }
}

//==============================================================================
// DeviceQueue default derivative implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// vjpMinMax
    @inlinable
    func vjpMinMax<T, R>(
        _ x: T, _ y: T, _ scale: T,
        _ op: @escaping (T.Element, T.Element) -> Bool,
        _ resultTrue: inout R, _ resultFalse: inout R)
        where
        T: ShapedBuffer, T.Element: Comparable & Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_vjpMinMax(x, y, scale, op, &resultTrue, &resultFalse)
    }
}

