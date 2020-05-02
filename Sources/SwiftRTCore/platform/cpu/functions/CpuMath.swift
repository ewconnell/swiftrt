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

public protocol CpuFunctions { }

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue & CpuMapOps {
    @inlinable func abs<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { Swift.abs($0) }
    }

    @inlinable
    func acos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .acos($0) }
    }

    @inlinable
    func acosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .acosh($0) }
    }

    @inlinable
    public func add<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                         _ result: inout Tensor<S,E>)
    where S: TensorShape, E: AdditiveArithmetic
    {
        lhs.read(using: self)
        rhs.read(using: self)
        result.readWrite(using: self)
        mapOp(lhs, rhs, &result, +)
    }

    @inlinable
    func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                _ result: inout Tensor<S,Bool>)
    where S: TensorShape
    {
        mapOp(lhs, rhs, &result) { $0 && $1 }
    }

    @inlinable
    func asin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .asin($0) }
    }

    @inlinable
    func asinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .asinh($0) }
    }

    @inlinable
    func atan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .atan($0) }
    }

    @inlinable
    func atan2<S,E>(_ y: Tensor<S,E>, _ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(y, x, &result) { .atan2(y: $0, x: $1) }
    }

    @inlinable
    func atanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .atanh($0) }
    }

    // FloatingPoint -> Integer
    @inlinable
    func cast<S, E, RE>(from buffer: Tensor<S,E>, to result: inout Tensor<S,RE>)
    where S: TensorShape, E: BinaryFloatingPoint, RE: BinaryInteger
    {
        mapOp(buffer, &result) { RE($0) }
    }

    // Integer -> FloatingPoint
    @inlinable
    func cast<S, E, RE>(from buffer: Tensor<S,E>, to result: inout Tensor<S,RE>)
    where S: TensorShape, E: BinaryInteger, RE: BinaryFloatingPoint
    {
        mapOp(buffer, &result) { RE($0) }
    }

    @inlinable
    func copy<S,E>(from x: Tensor<S,E>, to result: inout Tensor<S,E>)
    where S: TensorShape
    {
        x.read(using: self)
        result.readWrite(using: self)
        mapOp(x, &result) { $0 }
    }

    @inlinable
    func cos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .cos($0) }
    }

    @inlinable
    func cosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .cosh($0) }
    }

    @inlinable
    func delay(_ interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }

    @inlinable
    func div<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E: AlgebraicField
    {
        mapOp(lhs, rhs, &result, /)
    }

    @inlinable
    func elementsAlmostEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                  _ tolerance: E,
                                  _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: SignedNumeric & Comparable
    {
        mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
    }

    @inlinable
    func equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                    _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Equatable
    {
        mapOp(lhs, rhs, &result, ==)
    }

    @inlinable
    func erf<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .erf($0) }
    }

    @inlinable
    func erfc<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .erfc($0) }
    }

    @inlinable
    func exp<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp($0) }
    }

    @inlinable
    func exp2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp2($0) }
    }

    @inlinable
    func exp10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp10($0) }
    }

    @inlinable
    func expMinusOne<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .expMinusOne($0) }
    }

    @inlinable
    func gamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .gamma($0) }
    }

    @inlinable
    func greater<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                      _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, >)
    }

    @inlinable
    func greaterOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                             _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, >=)
    }

    @inlinable
    func hypot<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, y, &result) { .hypot($0, $1) }
    }

    @inlinable
    func less<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                   _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, <)
    }

    @inlinable
    func lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                          _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, <=)
    }

    @inlinable
    func log<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log($0) }
    }

    @inlinable
    func log<S,E>(onePlus x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log(onePlus: $0) }
    }

    @inlinable
    func log2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log2($0) }
    }

    @inlinable
    func log10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log10($0) }
    }

    @inlinable
    func logGamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .logGamma($0) }
    }

    @inlinable
    func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
    }

    @inlinable
    func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
    }

    @inlinable
    func mul<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Numeric
    {
        mapOp(lhs, rhs, &result, *)
    }

    @inlinable
    func neg<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: SignedNumeric
    {
        mapOp(x, &result, -)
    }

    @inlinable
    func notEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                       _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E: Equatable
    {
        mapOp(lhs, rhs, &result, !=)
    }

    @inlinable
    func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
               _ result: inout Tensor<S,Bool>)
    where S: TensorShape
    {
        mapOp(lhs, rhs, &result) { $0 || $1 }
    }

    @inlinable
    func pow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, y, &result) { .pow($0, $1) }
    }

    @inlinable
    func pow<S, E>(_ x: Tensor<S,E>, _ n: Int, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .pow($0, n) }
    }

    @inlinable
    func replace<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                      _ condition: Tensor<S,Bool>,
                      _ result: inout Tensor<S,E>)
    {
        mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
    }

    @inlinable
    func root<S,E>(_ x: Tensor<S,E>, _ n: Int, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .root($0, n) }
    }

    @inlinable
    func sign<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { $0 < 0 ? -1 : 1 }
    }

    @inlinable
    func signGamma<S,E>(_ x: Tensor<S,E>, _ result: inout FloatingPointSign)
    where S: TensorShape, E: Real
    {
        // TODO: don't know what to do with this as set operation
        fatalError("Not implemented")
    }

    @inlinable
    func sin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sin($0) }
    }

    @inlinable
    func sinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sinh($0) }
    }

    @inlinable
    func subtract<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                       _ result: inout Tensor<S,E>)
    where S: TensorShape, E: AdditiveArithmetic
    {
        mapOp(lhs, rhs, &result, -)
    }

    @inlinable
    func sqrt<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sqrt($0) }
    }

    //--------------------------------------------------------------------------
    @inlinable
    func squared<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Numeric
    {
        mapOp(x, &result) { $0 * $0 }
    }

    //--------------------------------------------------------------------------
    @inlinable
    func tan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .tan($0) }
    }
    
    @inlinable
    func tanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .tanh($0) }
    }

    //==========================================================================
    // specialized derivative implementations
    //==========================================================================
    /// vjpMinMax
    @inlinable func vjpMinMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ op: @escaping (E, E) -> Bool,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
    where S: TensorShape, E: Comparable & Numeric
    {
        mapOp(x, y, scale, &resultTrue, &resultFalse) {
            op($0, $1) ? ($2, E.zero) : (E.zero, $2)
        }
    }
}
