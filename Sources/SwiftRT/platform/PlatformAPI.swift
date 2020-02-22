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
import Real

//==============================================================================
// PlatformAPI
// This is the platform user application interface
//
public protocol PlatformAPI {
    // queue managment
    mutating func useCpu()
    mutating func use(device: Int, queue: Int)
    mutating func using<R>(device: Int, queue: Int, _ body: () -> R) -> R
    mutating func using<R>(queue: Int, _ body: () -> R) -> R
    
    //--------------------------------------------------------------------------
    /// retrieve the name of a buffer for diagnostics
    var memoryManager: MemoryManagement { get }
    
    func read<T>(_ tensor: T) -> ElementBuffer<T.Element, T.Shape>
        where T: TensorView

    func write<T>(_ tensor: inout T,
                  willOverwrite: Bool,
                  copyIfNotUniquelyReferenced: Bool,
                  copyIfNotDense: Bool)
        -> MutableElementBuffer<T.Element, T.Shape> where T: TensorView

    //--------------------------------------------------------------------------
    /// abs
    func abs<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    
    func absmax<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: SignedNumeric & Comparable
    
    func abssum<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: SignedNumeric & Comparable
    
    /// add
    func add<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    
    // all
    func all<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element == Bool

    /// and
    func and<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool

    // any
    func any<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element == Bool

    /// cast
    func cast<T, U>(_ other: U) -> T where
        T: TensorView, T.Element: AnyConvertable,
        U: TensorView, U.Element: AnyConvertable, U.Shape == T.Shape
    
    /// concat
    func concat<T>(_ tensors: [T], alongAxis axis: Int, _ name: String?) -> T
        where T: TensorView
    
    /// copy  performs an indexed copy
    func copy<T>(from x: T, to result: inout T) where T: TensorView
    
    /// delayQueue
    func delayQueue(atLeast interval: TimeInterval)
    
    /// div
    func div<T>(_ lhs: T, _ rhs: T) -> T where
        T: TensorView, T.Element: AlgebraicField
    
    /// elementsAlmostEqual
    func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T, tolerance: T.Element)
        -> T.BoolView
        where T: TensorView, T.Element: SignedNumeric & Comparable
    
    /// equal
    func equal<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView
    
    /// exp
    func exp<T>(_ x: T) -> T where
        T: TensorView, T.Element: Real
    
    /// fill(result:with element:
    func fill<T>(_ result: inout T, with element: T.Element)
        where T: TensorView
    
    /// fill(result:with range:
    func fill<T, R>(_ result: inout T, with range: R) where
        T: TensorView,
        R: StridedRangeExpression, R.Bound == T.Element
    
    /// greater
    func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    
    /// greaterOrEqual
    func greaterOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    
    /// less
    func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    
    /// lessOrEqual
    func lessOrEqual<T>(_ lhs: T, _ rhs: T) ->T.BoolView
        where T: TensorView, T.Element: Comparable
    
    /// log
    func log<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    
    /// Computes the element-wise maximum of two tensors.
    func max<T>(_ lhs: T, _ rhs: T) -> T
         where T: TensorView, T.Element: Comparable
    
    func max<T>(_ x: T, alongAxes axes: Set<Int>?) -> T
        where T: TensorView, T.Element: Comparable

    func mean<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: AlgebraicField

    /// Computes the element-wise minimum of two tensors.
    func min<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: Comparable
    
    func min<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: Comparable

    /// mul
    func mul<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: Numeric
    
    /// neg
    /// returns the element-wise negation of `x`
    func neg<T>(_ x: T) -> T
        where T: TensorView, T.Element: SignedNumeric
    
    /// notEqual
    func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView
    
    /// or
    func or<T>(_ lhs: T, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool
    
    /// pow
    func pow<T>(_ x: T, _ y: T) -> T
        where T: TensorView, T.Element: Real
    
    func prod<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: Numeric

    func prodNonZeros<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: Numeric

    /// replace
    func replace<T>(_ x: T, with y: T, where condition: T.BoolView) -> T
        where T: TensorView
    
    /// sign
    func sign<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    
    /// subtract
    func subtract<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    
    /// sqrt
    func sqrt<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    
    /// sqrtSumSquares
    func sqrtSumSquares<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: Real

    /// squared
    func squared<T>(_ x: T) -> T
        where T: TensorView, T.Element: Numeric
    
    func sum<T>(_ x: T, alongAxes: Set<Int>?) -> T
        where T: TensorView, T.Element: Numeric

    //==========================================================================
    // derivative function declarations
    
    func _vjpAbs<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    
    func _vjpAdd<T>(lhs: T, rhs: T) -> (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView

    func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where
        T: DifferentiableTensorView, T.Element: AlgebraicField & SignedNumeric

    func _vjpExp<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real

    func _vjpLog<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real

    func _vjpMinMax<T>(_ x: T, _ y: T, _ scale: T,
                       _ op: @escaping (T.Element, T.Element) -> Bool) -> (T, T)
        where T : TensorView, T.Element : Comparable, T.Element : Numeric
    
    func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensorView

    func _vjpNeg<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: SignedNumeric

    func _vjpPow<T>(_ x: T, _ y: T) -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Real

    func _vjpSign<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real

    func _vjpSqrt<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real

    func _vjpSquared<T>(_ x: T) -> (value: T, pullback: (T) -> (T))
        where T: DifferentiableTensorView

    func _vjpSubtract<T>(lhs: T, rhs: T) -> (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView

    func _vjpSum<T>(_ x: T, alongAxes: Set<Int>?)
        -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
}

//==============================================================================
// PlatformAPI extensions
public extension PlatformAPI {
    func write<T>(_ tensor: inout T, willOverwrite: Bool)
        -> MutableElementBuffer<T.Element, T.Shape> where T: TensorView
    {
        write(&tensor, willOverwrite: willOverwrite,
              copyIfNotUniquelyReferenced: true,
              copyIfNotDense: true)
    }
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

public typealias ReduceOpFinal<R: MutableShapedBuffer> = (R.Element) -> R.Element

//==============================================================================
// parameter matching helper
@inlinable
public func implicitlyMatchExtents<T>(_ lhs: T, _ rhs: T) -> (T, T)
    where T: TensorView
{
    if lhs.count == rhs.count {
        return (lhs, rhs)
    } else if lhs.count > rhs.count {
        return (lhs, rhs.repeated(to: lhs.extents))
    } else {
        return (lhs.repeated(to: rhs.extents), rhs)
    }
}

