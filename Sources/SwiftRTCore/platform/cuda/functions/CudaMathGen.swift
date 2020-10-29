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

import Numerics
import SwiftRTCuda

// gyb utility docs
// https://nshipster.com/swift-gyb/

//******************************************************************************
//
// DO NOT EDIT. THIS FILE IS GENERATED FROM .swift.gyb file
//
//******************************************************************************

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable func abs<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & SignedNumeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_abs(x, &out)
      return
    }
    diagnostic(.queueGpu, "abs() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAbs(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.abs(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func acos<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_acos(x, &out)
      return
    }
    diagnostic(.queueGpu, "acos() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAcos(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.acos(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func acosh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_acosh(x, &out)
      return
    }
    diagnostic(.queueGpu, "acosh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAcosh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.acosh(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func asin<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_asin(x, &out)
      return
    }
    diagnostic(.queueGpu, "asin() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAsin(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.asin(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func asinh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_asinh(x, &out)
      return
    }
    diagnostic(.queueGpu, "asinh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAsinh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.asinh(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func atan<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_atan(x, &out)
      return
    }
    diagnostic(.queueGpu, "atan() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAtan(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.atan(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func atanh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_atanh(x, &out)
      return
    }
    diagnostic(.queueGpu, "atanh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtAtanh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.atanh(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func cos<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_cos(x, &out)
      return
    }
    diagnostic(.queueGpu, "cos() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtCos(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.cos(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func cosh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_cosh(x, &out)
      return
    }
    diagnostic(.queueGpu, "cosh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtCosh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.cosh(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func erf<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_erf(x, &out)
      return
    }
    diagnostic(.queueGpu, "erf() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtErf(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.erf(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func erfc<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_erfc(x, &out)
      return
    }
    diagnostic(.queueGpu, "erfc() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtErfc(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.erfc(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func exp<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_exp(x, &out)
      return
    }
    diagnostic(.queueGpu, "exp() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtExp(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.exp(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func exp2<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_exp2(x, &out)
      return
    }
    diagnostic(.queueGpu, "exp2() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtExp2(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.exp2(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func exp10<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_exp10(x, &out)
      return
    }
    diagnostic(.queueGpu, "exp10() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtExp10(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.exp10(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func expMinusOne<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_expMinusOne(x, &out)
      return
    }
    diagnostic(.queueGpu, "expMinusOne() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtExpMinusOne(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.expMinusOne(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func gamma<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_gamma(x, &out)
      return
    }
    diagnostic(.queueGpu, "gamma() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtGamma(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.gamma(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func log<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_log(x, &out)
      return
    }
    diagnostic(.queueGpu, "log() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtLog(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.log(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func log2<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_log2(x, &out)
      return
    }
    diagnostic(.queueGpu, "log2() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtLog2(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.log2(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func log10<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_log10(x, &out)
      return
    }
    diagnostic(.queueGpu, "log10() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtLog10(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.log10(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func logGamma<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_logGamma(x, &out)
      return
    }
    diagnostic(.queueGpu, "logGamma() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtLogGamma(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.logGamma(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func neg<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: SignedNumeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_neg(x, &out)
      return
    }
    diagnostic(.queueGpu, "neg() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtNeg(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.neg(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func sigmoid<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sigmoid(x, &out)
      return
    }
    diagnostic(.queueGpu, "sigmoid() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSigmoid(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.sigmoid(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func sign<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & SignedNumeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sign(x, &out)
      return
    }
    diagnostic(.queueGpu, "sign() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSign(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.sign(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func sin<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sin(x, &out)
      return
    }
    diagnostic(.queueGpu, "sin() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSin(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.sin(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func sinh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sinh(x, &out)
      return
    }
    diagnostic(.queueGpu, "sinh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSinh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.sinh(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func sqrt<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sqrt(x, &out)
      return
    }
    diagnostic(.queueGpu, "sqrt() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSqrt(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.sqrt(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func squared<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_squared(x, &out)
      return
    }
    diagnostic(.queueGpu, "squared() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtSquared(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.squared(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func tan<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_tan(x, &out)
      return
    }
    diagnostic(.queueGpu, "tan() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtTan(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.tan(x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func tanh<S, E>(
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_tanh(x, &out)
      return
    }
    diagnostic(.queueGpu, "tanh() on \(name)", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtTanh(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.tanh(x, &out) }
  }

}
