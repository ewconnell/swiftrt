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
extension CpuFunctions where Self: DeviceQueue {
  //--------------------------------------------------------------------------
  @inlinable public func cpu_abssum<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: SignedNumeric & Comparable {
    diagnostic(.queueCpu, "abssum(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.zero) { $0 += Swift.abs($1) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_all<S>(
    _ a: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) {
    diagnostic(.queueCpu, "all(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, true) { $0 = $0 && $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_any<S>(
    _ a: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) {
    diagnostic(.queueCpu, "any(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, false) { $0 = $0 || $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_sum<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    diagnostic(.queueCpu, "sum(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.zero) { $0 += $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_mean<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    diagnostic(.queueCpu, "mean(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.zero) { $0 += $1 }

    // the reduction count is the product of the reduced dimensions
    var prod = a.count
    if out.count > 1 {
      prod = 1
      for i in 0..<S.rank where out.shape[i] == 1 { prod *= a.shape[i] }
    }
    let scale = 1 / E.Value(exactly: prod)!

    // inplace divide by count
    mapOp(&out) { $0 * scale }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_min<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    diagnostic(.queueCpu, "min(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.highest) { $0 = Swift.min($0, $1) }
  }
  
  @inlinable public func cpu_argmin<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    diagnostic(.queueCpu, "argmin(\(a.name), axis: \(axis)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &arg, &out, E.Value.highest) { $0 = $0.value <= $1.value ? $0 : $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_max<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    diagnostic(.queueCpu, "max(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.lowest) { $0 = $0 > $1 ? $0 : $1 }
  }

  @inlinable public func cpu_argmax<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    diagnostic(.queueCpu, "argmax(\(a.name), axis: \(axis)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &arg, &out, E.Value.lowest) { $0 = $0.value > $1.value ? $0 : $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_prod<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "prod(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.one) { $0 *= $1 }
  }

  //--------------------------------------------------------------------------
  @inlinable public func cpu_prodNonZeros<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    diagnostic(.queueCpu, "prodNonZeros(\(a.name), axis: \(axis ?? 0)) on \(name)", categories: .queueCpu)
    cpu_reduce(a, axis, &out, E.Value.one) { if $1 != 0 { $0 *= $1 } }
  }
}

//==============================================================================
// CpuQueue functions with default cpu delegation
extension CpuQueue {
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  @inlinable public func abssum<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: SignedNumeric & Comparable {
    cpu_abssum(a, axis, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func all<S>(
    _ a: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) { cpu_all(a, axis, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func any<S>(
    _ a: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) { cpu_any(a, axis, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func sum<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic { cpu_sum(a, axis, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func mean<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField { cpu_mean(a, axis, &out) }
  //--------------------------------------------------------------------------
  @inlinable public func min<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    cpu_min(a, axis, &out)
  }
  //----------------------------------------------------------------------------
  @inlinable public func argmin<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    cpu_argmin(a, axis, &arg, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func max<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    cpu_max(a, axis, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func argmax<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    cpu_argmax(a, axis, &arg, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func prod<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_prod(a, axis, &out)
  }
  //--------------------------------------------------------------------------
  @inlinable public func prodNonZeros<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    cpu_prodNonZeros(a, axis, &out)
  }
}
