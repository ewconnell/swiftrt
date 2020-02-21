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
public protocol DeviceFunctions {
    /// the thread that created this queue. Used to detect accidental access
    var creatorThread: Thread { get }
    
    //--------------------------------------------------------------------------
    /// abs
    func abs<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element

    /// add
    func add<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element

    /// and
    func and<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool

    /// cast
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: AnyConvertable,
        R: MutableShapedBuffer, R.Element: AnyConvertable

    /// concat
    func concat<T, R>(buffers: [T], alongAxis axis: Int, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element

    /// copy
    func copy<T, R>(from x: T, to result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element

    /// delay
    func delay(atLeast interval: TimeInterval)

    /// div
    func div<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AlgebraicField,
        R: MutableShapedBuffer, R.Element == T.Element

    /// elementsAlmostEqual
    func elementsAlmostEqual<T, R>(lhs: T, rhs: T, tolerance: T.Element,
                                result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
        R: MutableShapedBuffer, R.Element == Bool

    /// equal
    func equal<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == Bool

    /// exp
    func exp<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element

    /// fill(result:with element:
    func fill<Element, R>(result: inout R, with element: Element) where
        R: MutableShapedBuffer, R.Element == Element

    /// fill(result:with range:
    func fill<T, R>(result: inout R, with range: T) where
        T: StridedRangeExpression,
        R: MutableShapedBuffer, R.Element == T.Bound

    /// greater
    func greater<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// greaterOrEqual
    func greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// less
    func less<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// lessOrEqual
    func lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// log
    func log<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// Computes the element-wise maximum of two tensors.
    func max<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// Computes the element-wise minimum of two tensors.
    func min<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// mul
    func mul<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// notEqual
    func notEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// or
    func or<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    
    /// pow
    func pow<T, R>(x: T, y: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// replace
    func replace<T, C, R>(x: T, with y: T, where condition: C,
                          result: inout R) where
        T: ShapedBuffer,
        C: ShapedBuffer, C.Element == Bool,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// sign
    func sign<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// subtract
    func subtract<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// sqrt
    func sqrt<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    
    /// squared
    func squared<T, R>(x: T, result: inout R) where
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
    func reduce<T, R>(x: T,
                   into result: inout R,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    
    //==========================================================================
    // derivative function declarations
    
    /// vjpMinMax
    func vjpMinMax<T, R>(
        x: T, y: T, scale: T,
        op: @escaping (T.Element, T.Element) -> Bool,
        resultTrue: inout R, resultFalse: inout R)
        where
        T: ShapedBuffer, T.Element: Comparable & Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
}


//==============================================================================
// DeviceQueue default cpu delegating implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// abs
    func abs<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// add
    func add<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// and
    func and<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// cast
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: AnyConvertable,
        R: MutableShapedBuffer, R.Element: AnyConvertable
    {
        
    }

    /// concat
    func concat<T, R>(buffers: [T], alongAxis axis: Int, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// copy
    func copy<T, R>(from x: T, to result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }
    
    /// delay
    func delay(atLeast interval: TimeInterval)
    {
        
    }

    /// div
    func div<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AlgebraicField,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// elementsAlmostEqual
    func elementsAlmostEqual<T, R>(lhs: T, rhs: T, tolerance: T.Element,
                                result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// equal
    func equal<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// exp
    func exp<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// fill(result:with element:
    func fill<Element, R>(result: inout R, with element: Element) where
        R: MutableShapedBuffer, R.Element == Element
    {
        
    }

    /// fill(result:with range:
    func fill<T, R>(result: inout R, with range: T) where
        T: StridedRangeExpression,
        R: MutableShapedBuffer, R.Element == T.Bound
    {
        
    }

    /// greater
    func greater<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// greaterOrEqual
    func greaterOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// less
    func less<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// lessOrEqual
    func lessOrEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// log
    func log<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// Computes the element-wise maximum of two tensors.
    func max<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// Computes the element-wise minimum of two tensors.
    func min<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// mul
    func mul<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// neg
    /// returns the element-wise negation of `x`
    func neg<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// notEqual
    func notEqual<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// or
    func or<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        
    }

    /// pow
    func pow<T, R>(x: T, y: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// replace
    func replace<T, C, R>(x: T, with y: T, where condition: C,
                          result: inout R) where
        T: ShapedBuffer,
        C: ShapedBuffer, C.Element == Bool,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// sign
    func sign<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// subtract
    func subtract<T, R>(lhs: T, rhs: T, result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// sqrt
    func sqrt<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// squared
    func squared<T, R>(x: T, result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    /// reduce
    /// Reduces `x` along the specified axes
    /// - Parameter x: value tensor
    /// - Parameter result: contains the initial value of the result on entry
    ///  and then final reduction result on return
    /// - Parameter opNext: the operation to perform on pairs of elements
    /// - Parameter opFinal: the operation to perform on the final result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func reduce<T, R>(x: T,
                   into result: inout R,
                   opId: ReductionOp,
                   opNext: @escaping (T.Element, T.Element) -> T.Element,
                   opFinal: ReduceOpFinal<T>?) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }
}

//==============================================================================
// DeviceQueue default derivative implementations
public extension DeviceFunctions where Self: DeviceQueue {
    /// vjpMinMax
    func vjpMinMax<T, R>(
        x: T, y: T, scale: T,
        op: @escaping (T.Element, T.Element) -> Bool,
        resultTrue: inout R, resultFalse: inout R)
        where
        T: ShapedBuffer, T.Element: Comparable & Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        cpu_vjpMinMax(x: x, y: y, scale: scale, op: op,
                      resultTrue: &resultTrue, resultFalse: &resultFalse)
    }
}

