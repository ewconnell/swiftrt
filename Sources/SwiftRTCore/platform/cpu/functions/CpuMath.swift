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
// Cpu device queue function implementations
extension DeviceQueue {
  //--------------------------------------------------------------------------
  @inlinable public func cpu_kernel<S, AE, RE>(
    _ a: Tensor<S, AE>,
    _ out: inout Tensor<S, RE>,
    _ opName: String,
    _ op: @escaping (AE.Value, RE.Value) -> RE.Value
  ) {
    diagnostic(.queueCpu, "\(opName) on \(name)", categories: .queueCpu)
    mapOp(a, &out, op)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_abs<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & SignedNumeric {
    diagnostic(.queueCpu, "abs(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { abs($0) }
  }

  @inlinable public func cpu_abs<S, E>(
    _ x: Tensor<S, Complex<E>>,
    _ out: inout Tensor<S, E>
  ) where E == E.Value, E.Value: Comparable & SignedNumeric {
    diagnostic(.queueCpu, "abs(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { abs($0) }
  }

  @inlinable public func abs2<S, E>(
    _ x: Tensor<S, Complex<E>>,
    _ out: inout Tensor<S, E>
  ) where E == E.Value, E.Value: Comparable & SignedNumeric {
    diagnostic(.queueCpu, "abs2(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { SwiftRTCore.abs2($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_acos<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "acos(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .acos($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_acosh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "acosh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .acosh($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    diagnostic(
      .queueCpu, "add(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, +)
  }

  @inlinable public func cpu_add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    diagnostic(
      .queueCpu, "add(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, +)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_and<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value == Bool {
    diagnostic(
      .queueCpu, "and(\(lhs.name), \(rhs.name) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 && $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_asin<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "asin(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .asin($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_asinh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "asinh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .asinh($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_atan<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "atan(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .atan($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_atan2<S, E>(
    _ y: Tensor<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "atan2(y: \(y.name), x: \(x.name)) on \(name)",
      categories: .queueCpu)
    mapOp(y, x, &out) { .atan2(y: $0, x: $1) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_atanh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "atanh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .atanh($0) }
  }

  //--------------------------------------------------------------------------
  // FloatingPoint -> Integer
  @inlinable public func cpu_cast<S, E, RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  ) where E.Value: BinaryFloatingPoint, RE.Value: BinaryInteger {
    diagnostic(.queueCpu, "cast(\(a.name)) on \(name)", categories: .queueCpu)
    mapOp(a, &out) { RE.Value($0) }
  }

  // Integer -> FloatingPoint
  @inlinable public func cpu_cast<S, E, RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  ) where E.Value: BinaryInteger, RE.Value: BinaryFloatingPoint {
    diagnostic(.queueCpu, "cast(\(a.name)) on \(name)", categories: .queueCpu)
    mapOp(a, &out) { RE.Value($0) }
  }

  @inlinable public func cpu_cast<S,E>(
    from a: Tensor<S,Bool>,
    to out: inout Tensor<S,E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "cast(\(a.name)) on \(name)", categories: .queueCpu)
    mapOp(a, &out) { $0 ? E.Value.one : E.Value.zero }
  }

  @inlinable public func cpu_cast<S,E>(
    from a: Tensor<S,E>,
    to out: inout Tensor<S,Bool>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "cast(\(a.name)) on \(name)", categories: .queueCpu)
    mapOp(a, &out) { $0 != E.Value.zero }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_copy<S, E>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, E>
  ) where S: TensorShape {
    diagnostic(
      .queueCpu, "copy(form: \(a.name), to: \(out.name) on \(name)",
      categories: .queueCpu)
    mapOp(a, &out) { $0 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_cos<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "cos(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .cos($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_cosh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "cosh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .cosh($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    diagnostic(
      .queueCpu, "div(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, /)
  }

  @inlinable public func cpu_div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    diagnostic(
      .queueCpu, "div(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, /)
  }

  @inlinable public func cpu_div<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    diagnostic(
      .queueCpu, "div(\(lhs), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, /)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_elementsAlmostEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ tolerance: E.Value,
    _ out: inout Tensor<S, Bool>
  )
  where E.Value: SignedNumeric & Comparable {
    diagnostic(
      .queueCpu,
      "elementsAlmostEqual(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { Swift.abs($0 - $1) <= tolerance }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_equal<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Equatable {
    diagnostic(
      .queueCpu, "equal(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, ==)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_erf<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "erf(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .erf($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_erfc<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "erfc(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .erfc($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_exp<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "exp(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .exp($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_exp2<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "exp2(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .exp2($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_exp10<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "exp10(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .exp10($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_expMinusOne<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "expMinusOne(\(x.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, &out) { .expMinusOne($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_gamma<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "gamma(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .gamma($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "greater(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, >)
  }

  @inlinable public func cpu_greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "greater(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, >)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu,
      "greaterOrEqual(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, >=)
  }

  @inlinable public func cpu_greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "greaterOrEqual(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, >=)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_hypot<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "hypot(\(x.name), \(y.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, &out) { .hypot($0, $1) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "less(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, <)
  }

  @inlinable public func cpu_less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "less(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, <)
  }
  //--------------------------------------------------------------------------
  @inlinable public func cpu_lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "lessOrEqual(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, <=)
  }

  @inlinable public func cpu_lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "lessOrEqual(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, <=)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_log<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "log(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .log($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_log<S, E>(
    onePlus x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "log(onePlus: \(x.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, &out) { .log(onePlus: $0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_log2<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "log2(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .log2($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_log10<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "log10(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .log10($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_logGamma<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "logGamma(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .logGamma($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_matmul<E>(
    _ lhs: TensorR2<E>, _ transposeLhs: Bool,
    _ rhs: TensorR2<E>, _ transposeRhs: Bool,
    _ out: inout TensorR2<E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "matmul on \(name)", categories: .queueCpu)
    let lhs = transposeLhs ? lhs.t : lhs
    let rhs = transposeRhs ? rhs.t : rhs
    assert(
      out.shape[0] == lhs.shape[0] && out.shape[1] == rhs.shape[1],
      "matmul inner dimensions must be equal")
    //-------------------------------
    // simple place holder
    for r in 0..<out.shape[0] {
      let row = lhs[r, ...]
      for c in 0..<out.shape[1] {
        let col = rhs[..., c]
        out[r, c] = zip(row, col).reduce(into: 0) { $0 += $1.0 * $1.1 }
      }
    }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_matmul<E>(
    _ lhs: TensorR3<E>, _ transposeLhs: Bool,
    _ rhs: TensorR3<E>, _ transposeRhs: Bool,
    _ out: inout TensorR3<E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "matmul on \(name)", categories: .queueCpu)
    let lhs = transposeLhs ? lhs.t : lhs
    let rhs = transposeRhs ? rhs.t : rhs
    assert(
      out.shape[0] == lhs.shape[0] && out.shape[1] == lhs.shape[1] && out.shape[2] == rhs.shape[2],
      "matmul inner dimensions must be equal")
    //-------------------------------
    // simple place holder
    for n in 0..<out.shape[0] {
      for r in 0..<out.shape[1] {
        let row = lhs[n, r, ...]
        for c in 0..<out.shape[2] {
          let col = rhs[n, ..., c]
          out[n, r, c] = zip(row, col).reduce(into: 0) { $0 += $1.0 * $1.1 }
        }
      }
    }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "max(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 >= $1 ? $0 : $1 }
  }

  @inlinable public func cpu_max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "max(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 >= $1 ? $0 : $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "min(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 < $1 ? $0 : $1 }
  }

  @inlinable public func cpu_min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    diagnostic(
      .queueCpu, "min(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 < $1 ? $0 : $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(
      .queueCpu, "mul(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, *)
  }

  @inlinable public func cpu_mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(
      .queueCpu, "mul(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, *)
  }

  //--------------------------------------------------------------------------
  // fused multiply add
  @inlinable public func cpu_multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(
      .queueCpu,
      "multiply(\(lhs.name), \(rhs.name), add: \(bias)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, bias, &out) { $0 * $1 + $2 }
  }

  @inlinable public func cpu_multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(
      .queueCpu,
      "multiply(\(lhs.name), \(rhs.name), add: \(bias.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, bias, &out) { $0 * $1 + $2 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_neg<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: SignedNumeric {
    diagnostic(.queueCpu, "neg(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out, -)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_notEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Equatable {
    diagnostic(
      .queueCpu, "notEqual(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, !=)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_or<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value == Bool {
    diagnostic(
      .queueCpu, "or(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out) { $0 || $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_pow<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "pow(x: \(x.name), y: \(y.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, &out) { .pow($0, $1) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_pow<S, E>(
    _ x: Tensor<S, E>,
    _ n: Int,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "pow(x: \(x.name), n: \(n)) on \(name)",
      categories: .queueCpu)
    mapOp(x, &out) { .pow($0, n) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_replace<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ condition: Tensor<S, Bool>,
    _ out: inout Tensor<S, E>
  ) {
    diagnostic(
      .queueCpu,
      "replace(x: \(x.name), y: \(y.name), " + "condition: \(condition.name)) on \(name)",
      categories: .queueCpu)
    mapOp(condition, y, x, &out) { $0 ? $1 : $2 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_root<S, E>(
    _ x: Tensor<S, E>,
    _ n: Int,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "root(x: \(x.name), n: \(n)) on \(name)",
      categories: .queueCpu)
    mapOp(x, &out) { .root($0, n) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sigmoid<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(
      .queueCpu, "sigmoid(\(x.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, &out) { 1 / (1 + .exp(-$0)) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sign<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & SignedNumeric {
    diagnostic(.queueCpu, "sign(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { $0 < 0 ? -1 : 1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sin<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "sin(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .sin($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sinh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "sinh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .sinh($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: AdditiveArithmetic {
    diagnostic(
      .queueCpu, "subtract(\(lhs.name), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, -)
  }

  @inlinable public func cpu_subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  )
  where E.Value: AdditiveArithmetic {
    diagnostic(
      .queueCpu, "subtract(\(lhs.name), \(rhs)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, -)
  }

  @inlinable public func cpu_subtract<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    diagnostic(
      .queueCpu, "subtract(\(lhs), \(rhs.name)) on \(name)",
      categories: .queueCpu)
    mapOp(lhs, rhs, &out, -)
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sqrt<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "sqrt(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .sqrt($0) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_squared<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "squared(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { $0 * $0 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_tan<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "tan(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .tan($0) }
  }

  @inlinable public func cpu_tanh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    diagnostic(.queueCpu, "tanh(\(x.name)) on \(name)", categories: .queueCpu)
    mapOp(x, &out) { .tanh($0) }
  }

  //==========================================================================
  // specialized derivative implementations
  //==========================================================================
  /// cpu_vjpMin
  @inlinable public func cpu_vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMin(x: \(x.name), y: \(y.name), " + "scale: \(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, scale, &out) { $0 <= $1 ? $2 : E.Value.zero }
  }

  /// cpu_vjpMin
  @inlinable public func cpu_vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMin(x: \(x.name), y: \(y), " + "scale: \(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, scale, y, &out) { $0 <= $2 ? $1 : E.Value.zero }
  }

  /// cpu_vjpMin
  @inlinable public func cpu_vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>,
    _ resultFalse: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMin(x: \(x.name), y: \(y.name), " + "scale: \(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, scale, &resultTrue, &resultFalse) {
      $0 <= $1 ? ($2, E.Value.zero) : (E.Value.zero, $2)
    }
  }

  /// cpu_vjpMin
  @inlinable public func cpu_vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>,
    _ resultFalse: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMin(x: \(x.name), y: \(y), scale: " + "\(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, scale, y, &resultTrue, &resultFalse) {
      $0 <= $2 ? ($1, E.Value.zero) : (E.Value.zero, $1)
    }
  }

  //--------------------------------------------------------------------------
  /// cpu_vjpMax
  @inlinable public func cpu_vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMax(x: \(x.name), y: \(y.name), scale: " + "\(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, scale, &out) { $0 >= $1 ? $2 : E.Value.zero }
  }

  /// cpu_vjpMax
  @inlinable public func cpu_vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMax(x: \(x.name), y: \(y), scale: " + "\(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, scale, y, &out) { $0 >= $2 ? $1 : E.Value.zero }
  }

  /// cpu_vjpMax
  @inlinable public func cpu_vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>,
    _ resultFalse: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMax(x: \(x.name), y: \(y.name), scale: " + "\(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, y, scale, &resultTrue, &resultFalse) {
      $0 >= $1 ? ($2, E.Value.zero) : (E.Value.zero, $2)
    }
  }

  /// cpu_vjpMax
  @inlinable public func cpu_vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>,
    _ resultFalse: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    diagnostic(
      .queueCpu, "vjpMax(x: \(x.name), y: \(y), scale: " + "\(scale.name)) on \(name)",
      categories: .queueCpu)
    mapOp(x, scale, y, &resultTrue, &resultFalse) {
      $0 >= $2 ? ($1, E.Value.zero) : (E.Value.zero, $1)
    }
  }
}

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CpuQueue {
  @inlinable public func kernel<S, AE, RE>(
    _ a: Tensor<S, AE>,
    _ out: inout Tensor<S, RE>,
    _ opName: String,
    _ op: @escaping (AE.Value, RE.Value) -> RE.Value
  ) {
    cpu_kernel(a, &out, opName, op)
  }

  //--------------------------------------------------------------------------
  @inlinable public func abs<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & SignedNumeric {
    cpu_abs(x, &out)
  }

  @inlinable public func abs<S, E>(
    _ x: Tensor<S, Complex<E>>,
    _ out: inout Tensor<S, E>
  ) where E == E.Value, E.Value: Comparable & SignedNumeric {
    cpu_abs(x, &out)
  }

  @inlinable public func abs2<S, E>(
    _ x: Tensor<S, Complex<E>>,
    _ out: inout Tensor<S, E>
  ) where E == E.Value, E.Value: Comparable & SignedNumeric {
    cpu_abs(x, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func acos<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    cpu_acos(x, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func acosh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_acosh(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    cpu_add(lhs, rhs, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    cpu_add(lhs, rhs, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func and<S, E>(
    _ lhs: Tensor<S, E>, _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value == Bool { cpu_and(lhs, rhs, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func asin<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_asin(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func asinh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_asinh(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func atan<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_atan(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func atan2<S, E>(
    _ y: Tensor<S, E>, _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Real { cpu_atan2(y, x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func atanh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_atanh(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func cast<S,E,RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  ) where E.Value: BinaryFloatingPoint, RE.Value: BinaryInteger {
    cpu_cast(from: a, to: &out)
  }
  
  @inlinable public func cast<S,E,RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  ) where E.Value: BinaryInteger, RE.Value: BinaryFloatingPoint {
    cpu_cast(from: a, to: &out)
  }

  @inlinable public func cast<S,E>(
    from a: Tensor<S,Bool>,
    to out: inout Tensor<S,E>
  ) where E.Value: Numeric {
    cpu_cast(from: a, to: &out)
  }

  @inlinable public func cast<S,E>(
    from a: Tensor<S,E>,
    to out: inout Tensor<S,Bool>
  ) where E.Value: Numeric {
    cpu_cast(from: a, to: &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func copy<S, E>(from a: Tensor<S, E>, to b: inout Tensor<S, E>)
  where S: TensorShape { cpu_copy(from: a, to: &b) }
  //--------------------------------------------------------------------------
  @inlinable public func cos<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_cos(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func cosh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_cosh(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    cpu_div(lhs, rhs, &out)
  }

  @inlinable public func div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    cpu_div(lhs, rhs, &out)
  }

  @inlinable public func div<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    cpu_div(lhs, rhs, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func elementsAlmostEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ tolerance: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: SignedNumeric & Comparable {
    cpu_elementsAlmostEqual(lhs, rhs, tolerance, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func equal<S, E>(
    _ lhs: Tensor<S, E>, _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  )
  where E.Value: Equatable { cpu_equal(lhs, rhs, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func erf<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_erf(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func erfc<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_erfc(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func exp<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_exp(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func exp2<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_exp2(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func exp10<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_exp10(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func expMinusOne<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_expMinusOne(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func gamma<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_gamma(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_greater(lhs, rhs, &out) }

  @inlinable public func greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_greater(lhs, rhs, &out) }

  //--------------------------------------------------------------------------
  @inlinable public func greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    cpu_greaterOrEqual(lhs, rhs, &out)
  }

  @inlinable public func greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    cpu_greaterOrEqual(lhs, rhs, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func hypot<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Real { cpu_hypot(x, y, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_less(lhs, rhs, &out) }

  @inlinable public func less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_less(lhs, rhs, &out) }

  //--------------------------------------------------------------------------
  @inlinable public func lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_lessOrEqual(lhs, rhs, &out) }

  @inlinable public func lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable { cpu_lessOrEqual(lhs, rhs, &out) }

  //--------------------------------------------------------------------------
  @inlinable public func log<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_log(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func log<S, E>(onePlus x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_log(onePlus: x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func log2<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_log2(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func log10<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_log10(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func logGamma<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_logGamma(x, &out) }
  //--------------------------------------------------------------------------
  //    @inlinable public func matmul2<E>(type: E.Type) -> DeviceMatmul2<E>
  //    where E: StorageElement, E.Value: StorageElement & Numeric { CpuMatmul2<E>() }
  //--------------------------------------------------------------------------
  @inlinable public func matmul<E>(
    _ lhs: TensorR2<E>, _ transposeLhs: Bool,
    _ rhs: TensorR2<E>, _ transposeRhs: Bool,
    _ out: inout TensorR2<E>
  ) where E.Value: Numeric {
    cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func matmul<E>(
    _ lhs: TensorR3<E>, _ transposeLhs: Bool,
    _ rhs: TensorR3<E>, _ transposeRhs: Bool,
    _ out: inout TensorR3<E>
  ) where E.Value: Numeric {
    cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    cpu_min(lhs, rhs, &out)
  }

  @inlinable public func min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    cpu_min(lhs, rhs, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    cpu_max(lhs, rhs, &out)
  }

  @inlinable public func max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    cpu_max(lhs, rhs, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_mul(lhs, rhs, &out)
  }

  @inlinable public func mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_mul(lhs, rhs, &out)
  }

  //--------------------------------------------------------------------------
  // fused multiply add
  @inlinable public func multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_multiply(lhs, rhs, add: bias, &out)
  }

  @inlinable public func multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_multiply(lhs, rhs, add: bias, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func neg<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: SignedNumeric { cpu_neg(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func notEqual<S, E>(
    _ lhs: Tensor<S, E>, _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  )
  where E.Value: Equatable { cpu_notEqual(lhs, rhs, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func or<S, E>(
    _ lhs: Tensor<S, E>, _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value == Bool { cpu_or(lhs, rhs, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func pow<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Real { cpu_pow(x, y, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func pow<S, E>(
    _ x: Tensor<S, E>, _ n: Int,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Real { cpu_pow(x, n, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func replace<S, E>(
    _ x: Tensor<S, E>, 
    _ y: Tensor<S, E>,
    _ condition: Tensor<S, Bool>,
    _ out: inout Tensor<S, E>
  ) { cpu_replace(x, y, condition, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func root<S, E>(
    _ x: Tensor<S, E>, _ n: Int,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Real { cpu_root(x, n, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func sigmoid<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_sigmoid(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func sign<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Comparable & SignedNumeric {
    cpu_sign(x, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func sin<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_sin(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func sinh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_sinh(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    cpu_subtract(lhs, rhs, &out)
  }

  @inlinable public func subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    cpu_subtract(lhs, rhs, &out)
  }

  @inlinable public func subtract<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    cpu_subtract(lhs, rhs, &out)
  }

  //--------------------------------------------------------------------------
  @inlinable public func sqrt<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_sqrt(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func squared<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Numeric { cpu_squared(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func tan<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_tan(x, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func tanh<S, E>(_ x: Tensor<S, E>, _ out: inout Tensor<S, E>)
  where E.Value: Real { cpu_tanh(x, &out) }
}

//==============================================================================
// DeviceQueue specialized derivative delegation
extension DeviceQueue where Self: CpuFunctions {
  //--------------------------------------------------------------------------
  @inlinable public func vjpMin<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>, _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMin(x, y, scale, &out) }

  @inlinable public func vjpMin<S, E>(
    _ x: Tensor<S, E>, _ y: E.Value, _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMin(x, y, scale, &out) }

  @inlinable public func vjpMin<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>, _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>, _ resultFalse: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMin(x, y, scale, &resultTrue, &resultFalse) }

  @inlinable public func vjpMin<S, E>(
    _ x: Tensor<S, E>, _ y: E.Value, _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>, _ resultFalse: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMin(x, y, scale, &resultTrue, &resultFalse) }

  //--------------------------------------------------------------------------
  @inlinable public func vjpMax<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>, _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMax(x, y, scale, &out) }

  @inlinable public func vjpMax<S, E>(
    _ x: Tensor<S, E>, _ y: E.Value, _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMax(x, y, scale, &out) }

  @inlinable public func vjpMax<S, E>(
    _ x: Tensor<S, E>, _ y: Tensor<S, E>, _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>, _ resultFalse: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMax(x, y, scale, &resultTrue, &resultFalse) }

  @inlinable public func vjpMax<S, E>(
    _ x: Tensor<S, E>, _ y: E.Value, _ scale: Tensor<S, E>,
    _ resultTrue: inout Tensor<S, E>, _ resultFalse: inout Tensor<S, E>
  )
  where E.Value: Comparable & Numeric { cpu_vjpMax(x, y, scale, &resultTrue, &resultFalse) }
}
