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
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions
{
    //--------------------------------------------------------------------------
    @inlinable func abs<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_abs(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func acos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_acos(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func acosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_acosh(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func add<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where S: TensorShape, E.Value: AdditiveArithmetic {
        cpu_add(lhs, rhs, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                           _ result: inout Tensor<S,Bool>)
    where S: TensorShape { cpu_and(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func asin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_asin(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func asinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_asinh(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func atan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_atan(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func atan2<S,E>(_ y: Tensor<S,E>, _ x: Tensor<S,E>,
                               _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_atan2(y, x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func atanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_atanh(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func cast<S, E, RE>(from buffer: Tensor<S,E>,
                                   to result: inout Tensor<S,RE>)
    where S: TensorShape, E.Value: BinaryFloatingPoint, RE.Value: BinaryInteger
    { cpu_cast(from: buffer, to: &result) }
    //--------------------------------------------------------------------------
    @inlinable func cast<S, E, RE>(from buffer: Tensor<S,E>,
                                   to result: inout Tensor<S,RE>)
    where S: TensorShape, E.Value: BinaryInteger, RE.Value: BinaryFloatingPoint
    { cpu_cast(from: buffer, to: &result) }
    //--------------------------------------------------------------------------
    @inlinable func copy<S,E>(from x: Tensor<S,E>, to result: inout Tensor<S,E>)
    where S: TensorShape { cpu_copy(from: x, to: &result) }
    //--------------------------------------------------------------------------
    @inlinable func cos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_cos(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func cosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_cosh(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func delay(_ interval: TimeInterval) { cpu_delay(interval) }
    //--------------------------------------------------------------------------
    @inlinable func div<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                             _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: AlgebraicField { cpu_div(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func elementsAlmostEqual<S,E>(
        _ lhs: Tensor<S,E>,
        _ rhs: Tensor<S,E>,
        _ tolerance: E.Value,
        _ result: inout Tensor<S,Bool>
    ) where S: TensorShape, E.Value: SignedNumeric & Comparable {
        cpu_elementsAlmostEqual(lhs, rhs, tolerance, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                               _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Equatable { cpu_equal(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func erf<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_erf(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func erfc<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_erfc(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func exp<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_exp(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func exp2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_exp2(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func exp10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_exp10(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func expMinusOne<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_expMinusOne(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func gamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_gamma(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func greater<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable { cpu_greater(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func greaterOrEqual<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,Bool>
    ) where S: TensorShape, E.Value: Comparable {
        cpu_greaterOrEqual(lhs, rhs, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func hypot<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                               _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_hypot(x, y, &result) }
    //--------------------------------------------------------------------------
    @inlinable func less<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                              _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable { cpu_less(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                     _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable { cpu_lessOrEqual(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func log<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_log(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func log<S,E>(onePlus x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_log(onePlus: x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func log2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_log2(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func log10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_log10(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func logGamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_logGamma(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func matmul<E>(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) where E.Value: Numeric {
        cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func matmul<E>(
        _ lhs: TensorR3<E>, _ transposeLhs: Bool,
        _ rhs: TensorR3<E>, _ transposeRhs: Bool,
        _ result: inout TensorR3<E>
    ) where E.Value: Numeric {
        cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                             _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable { cpu_max(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                             _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable { cpu_min(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func mul<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                             _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Numeric { cpu_mul(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func neg<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: SignedNumeric { cpu_neg(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func notEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                  _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Equatable { cpu_notEqual(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                          _ result: inout Tensor<S,Bool>)
    where S: TensorShape { cpu_or(lhs, rhs, &result) }
    //--------------------------------------------------------------------------
    @inlinable func pow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                             _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_pow(x, y, &result) }
    //--------------------------------------------------------------------------
    @inlinable func pow<S, E>(_ x: Tensor<S,E>, _ n: Int,
                              _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_pow(x, n, &result) }
    //--------------------------------------------------------------------------
    @inlinable func replace<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>,
        _ condition: Tensor<S,Bool>,
        _ result: inout Tensor<S,E>)
    { cpu_replace(x, y, condition, &result) }
    //--------------------------------------------------------------------------
    @inlinable func root<S,E>(_ x: Tensor<S,E>, _ n: Int,
                              _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_root(x, n, &result) }
    //--------------------------------------------------------------------------
    @inlinable func sigmoid<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_sigmoid(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func sign<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_sign(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func signGamma<S,E>(_ x: Tensor<S,E>,
                                   _ result: inout FloatingPointSign)
    where S: TensorShape, E.Value: Real { cpu_signGamma(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func sin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_sin(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func sinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_sinh(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func subtract<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where S: TensorShape, E.Value: AdditiveArithmetic {
        cpu_subtract(lhs,rhs,&result)
    }
    //--------------------------------------------------------------------------
    @inlinable func sqrt<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_sqrt(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func squared<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Numeric { cpu_squared(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func tan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_tan(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func tanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real { cpu_tanh(x, &result) }
}

//==============================================================================
// DeviceQueue specialized derivative delegation
extension DeviceQueue where Self: CpuFunctions
{
    //--------------------------------------------------------------------------
    @inlinable func vjpMin<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric
    { cpu_vjpMin(x, y, scale, &result) }

    @inlinable func vjpMin<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric
    { cpu_vjpMin(x, y, scale, &resultTrue, &resultFalse) }
    
    //--------------------------------------------------------------------------
    @inlinable func vjpMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric
    { cpu_vjpMax(x, y, scale, &result) }

    @inlinable func vjpMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric
    { cpu_vjpMax(x, y, scale, &resultTrue, &resultFalse) }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue
{
    //--------------------------------------------------------------------------
    @inlinable func cpu_abs<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { Swift.abs($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_acos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .acos($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_acosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .acosh($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_add<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: AdditiveArithmetic {
        mapOp(lhs, rhs, &result, +)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                               _ result: inout Tensor<S,Bool>)
    where S: TensorShape {
        mapOp(lhs, rhs, &result) { $0 && $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_asin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .asin($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_asinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .asinh($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_atan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .atan($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_atan2<S,E>(_ y: Tensor<S,E>, _ x: Tensor<S,E>,
                                   _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(y, x, &result) { .atan2(y: $0, x: $1) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_atanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .atanh($0) }
    }
    
    //--------------------------------------------------------------------------
    // FloatingPoint -> Integer
    @inlinable func cpu_cast<S, E, RE>(
        from buffer: Tensor<S,E>,
        to result: inout Tensor<S,RE>
    ) where S: TensorShape, E.Value: BinaryFloatingPoint,
            RE.Value: BinaryInteger {
        mapOp(buffer, &result) { RE.Value($0) }
    }
    
    //--------------------------------------------------------------------------
    // Integer -> FloatingPoint
    @inlinable func cpu_cast<S, E, RE>(
        from buffer: Tensor<S,E>,
        to result: inout Tensor<S,RE>
    ) where S: TensorShape, E.Value: BinaryInteger,
            RE.Value: BinaryFloatingPoint {
        mapOp(buffer, &result) { RE.Value($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_copy<S,E>(from x: Tensor<S,E>, to result: inout Tensor<S,E>)
    where S: TensorShape {
        x.read(using: self)
        result.readWrite(using: self)
        mapOp(x, &result) { $0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_cos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .cos($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_cosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .cosh($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_delay(_ interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_div<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: AlgebraicField {
        mapOp(lhs, rhs, &result, /)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_elementsAlmostEqual<S,E>(
        _ lhs: Tensor<S,E>,
        _ rhs: Tensor<S,E>,
        _ tolerance: E.Value,
        _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: SignedNumeric & Comparable {
        mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                   _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Equatable {
        mapOp(lhs, rhs, &result, ==)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_erf<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .erf($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_erfc<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .erfc($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_exp<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .exp($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_exp2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .exp2($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_exp10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .exp10($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_expMinusOne<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .expMinusOne($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_gamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .gamma($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_greater<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                     _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result, >)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_greaterOrEqual<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result, >=)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_hypot<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                                   _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, y, &result) { .hypot($0, $1) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_less<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                  _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result, <)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                         _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result, <=)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_log<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .log($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_log<S,E>(onePlus x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .log(onePlus: $0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_log2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .log2($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_log10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .log10($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_logGamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .logGamma($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_matmul<E>(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) where E.Value: Numeric {
        let lhs = transposeLhs ? lhs.t : lhs
        let rhs = transposeRhs ? rhs.t : rhs
        assert(result.shape[0] == lhs.shape[0] &&
                result.shape[1] == rhs.shape[1],
               "matmul inner dimensions must be equal")
        //-------------------------------
        // simple place holder
        for r in 0..<result.shape[0] {
            let row = lhs[r, ...]
            for c in 0..<result.shape[1] {
                let col = rhs[..., c]
                result[r, c] = zip(row, col).reduce(into: 0) { $0 += $1.0 * $1.1 }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_matmul<E>(
        _ lhs: TensorR3<E>, _ transposeLhs: Bool,
        _ rhs: TensorR3<E>, _ transposeRhs: Bool,
        _ result: inout TensorR3<E>
    ) where E.Value: Numeric {
        let lhs = transposeLhs ? lhs.t : lhs
        let rhs = transposeRhs ? rhs.t : rhs
        assert(result.shape[0] == lhs.shape[0] &&
                result.shape[1] == lhs.shape[1] &&
                result.shape[2] == rhs.shape[2],
               "matmul inner dimensions must be equal")
        //-------------------------------
        // simple place holder
        for n in 0..<result.shape[0] {
            for r in 0..<result.shape[1] {
                let row = lhs[n, r, ...]
                for c in 0..<result.shape[2] {
                    let col = rhs[n, ..., c]
                    result[n, r, c] = zip(row, col).reduce(into: 0) { $0 += $1.0 * $1.1 }
                }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable {
        mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_mul<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Numeric {
        mapOp(lhs, rhs, &result, *)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_neg<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: SignedNumeric {
        mapOp(x, &result, -)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_notEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                      _ result: inout Tensor<S,Bool>)
    where S: TensorShape, E.Value: Equatable {
        mapOp(lhs, rhs, &result, !=)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                              _ result: inout Tensor<S,Bool>)
    where S: TensorShape {
        mapOp(lhs, rhs, &result) { $0 || $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_pow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                                 _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, y, &result) { .pow($0, $1) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_pow<S, E>(_ x: Tensor<S,E>, _ n: Int,
                                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .pow($0, n) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_replace<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>,
        _ condition: Tensor<S,Bool>,
        _ result: inout Tensor<S,E>)
    {
        mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_root<S,E>(_ x: Tensor<S,E>, _ n: Int,
                                  _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .root($0, n) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_sigmoid<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { 1 / (1 + .exp(-$0)) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_sign<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { $0 < 0 ? -1 : 1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_signGamma<S,E>(_ x: Tensor<S,E>,
                                       _ result: inout FloatingPointSign)
    where S: TensorShape, E.Value: Real {
        // TODO: don't know what to do with this as set operation
        fatalError("Not implemented")
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_sin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .sin($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_sinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .sinh($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_subtract<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: AdditiveArithmetic {
        mapOp(lhs, rhs, &result, -)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_sqrt<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .sqrt($0) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_squared<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Numeric {
        mapOp(x, &result) { $0 * $0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_tan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .tan($0) }
    }
    
    @inlinable func cpu_tanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Real {
        mapOp(x, &result) { .tanh($0) }
    }
    
    //==========================================================================
    // specialized derivative implementations
    //==========================================================================
    /// cpu_vjpMin
    @inlinable func cpu_vjpMin<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric {
        mapOp(x, y, scale, &result) { $0 <= $1 ? $2 : E.Value.zero }
    }
    /// cpu_vjpMin
    @inlinable func cpu_vjpMin<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric {
        mapOp(x, y, scale, &resultTrue, &resultFalse) {
            $0 <= $1 ? ($2, E.Value.zero) : (E.Value.zero, $2)
        }
    }
    
    //--------------------------------------------------------------------------
    /// cpu_vjpMax
    @inlinable func cpu_vjpMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ result: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric {
        mapOp(x, y, scale, &result) { $0 >= $1 ? $2 : E.Value.zero }
    }
    /// cpu_vjpMax
    @inlinable func cpu_vjpMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
    where S: TensorShape, E.Value: Comparable & Numeric {
        mapOp(x, y, scale, &resultTrue, &resultFalse) {
            $0 >= $1 ? ($2, E.Value.zero) : (E.Value.zero, $2)
        }
    }
}
