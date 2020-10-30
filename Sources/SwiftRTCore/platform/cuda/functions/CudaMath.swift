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

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable func abs<S, E>(
    _ x: Tensor<S, Complex<E>>,
    _ out: inout Tensor<S, E>
  ) where E == E.Value, E.Value: Comparable & SignedNumeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_abs(x, &out)
      return
    }
    diagnostic(.queueGpu, "abs(\(x.name)) on \(name)", categories: .queueGpu)
    var status: cudaError_t

    if x.order == out.order && x.isContiguous && out.isContiguous {
      status = srtAbsFlat(
        Complex<E>.type,
        x.deviceRead(using: self),
        E.type,
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )
    } else {
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { xData, x in
          srtAbs(xData, x, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.abs(x, &out) }
  }

  //--------------------------------------------------------------------------
  // tensor tensor
  @inlinable public func add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_add(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "add(\(lhs.name), \(rhs.name))", categories: .queueGpu)
    var status: cudaError_t

    if canFlatten(lhs, rhs, out) {
      status = srtAddFlat(
        E.type,
        lhs.deviceRead(using: self),
        rhs.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )

    } else {
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            srtAdd(l, lDesc, r, rDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.add(lhs, rhs, &out) }
  }

  //----------------------------------
  // add tensor Element
  @inlinable public func add<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_add(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "add(\(lhs.name), \(rhs))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtAddTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.add(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // div tensor tensor
  @inlinable public func div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_div(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "div(\(lhs.name), \(rhs.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtDiv(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.div(lhs, rhs, &out) }
  }

  //----------------------------------
  // div tensor Element
  @inlinable public func div<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_div(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "div(\(lhs.name), \(rhs))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtDivTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.div(lhs, rhs, &out) }
  }

  //----------------------------------
  // div Element tensor
  @inlinable public func div<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_div(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "div(\(lhs), \(rhs.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: lhs) { l in
        rhs.withTensor(using: self) { r, rDesc in
          srtDivET(l, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.div(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // mul tensor tensor
  @inlinable public func mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_mul(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "mul(\(lhs.name), \(rhs.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtMul(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.mul(lhs, rhs, &out) }
  }

  //----------------------------------
  // mul tensor Element
  @inlinable public func mul<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_mul(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "mul(\(lhs.name), \(rhs))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtMulTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.mul(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // subtract tensor tensor
  @inlinable public func subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_subtract(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "subtract(\(lhs.name), \(rhs.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtSub(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.subtract(lhs, rhs, &out) }
  }

  //----------------------------------
  // subtract tensor Element
  @inlinable public func subtract<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_subtract(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "subtract(\(lhs.name), \(rhs))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtSubTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.subtract(lhs, rhs, &out) }
  }

  //----------------------------------
  // subtract tensor tensor
  @inlinable public func subtract<S, E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_subtract(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "subtract(\(lhs), \(rhs.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: lhs) { l in
        rhs.withTensor(using: self) { r, rDesc in
          srtSubET(l, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.subtract(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // fused multiply add

  // multiply tensor tensor tensor
  @inlinable func multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(
      lhs.order == rhs.order && lhs.order == bias.order,
      _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_multiply(lhs, rhs, add: bias, &out)
      return
    }
    diagnostic(
      .queueGpu, "multiply(\(lhs.name), \(rhs.name), add: \(bias.name))",
      categories: .queueGpu)
    var status: cudaError_t

    if canFlatten(lhs, rhs, bias, out) {
      status = srtMultiplyAddFlat(
        E.type,
        lhs.deviceRead(using: self),
        rhs.deviceRead(using: self),
        bias.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream)

    } else {
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            bias.withTensor(using: self) { b, bDesc in
              srtMultiplyAdd(l, lDesc, r, rDesc, b, bDesc, o, oDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.multiply(lhs, rhs, add: bias, &out) }
  }

  //----------------------------------
  // multiply tensor tensor Element
  @inlinable func multiply<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    add bias: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_multiply(lhs, rhs, add: bias, &out)
      return
    }
    diagnostic(.queueGpu, "multiply(\(lhs.name), \(rhs.name), add: \(bias))", categories: .queueGpu)
    var status: cudaError_t

    if canFlatten(lhs, rhs, out) {
      status = withUnsafePointer(to: bias) { pbias in
        srtMultiplyAddFlatTTE(
          E.type,
          lhs.deviceRead(using: self),
          rhs.deviceRead(using: self),
          pbias,
          out.deviceReadWrite(using: self),
          out.count,
          stream)
      }
    } else {
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            withUnsafePointer(to: bias) { b in
              srtMultiplyAddTTE(l, lDesc, r, rDesc, b, o, oDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.multiply(lhs, rhs, add: bias, &out) }
  }
}

//==============================================================================
// Additional math ops with unique arguments
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable func atan2<S, E>(
    _ y: Tensor<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_atan2(y, x, &out)
      return
    }
    diagnostic(.queueGpu, "atan2(y: \(y.name), x: \(x.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      y.withTensor(using: self) { yData, y in
        x.withTensor(using: self) { xData, x in
          srtAtan2(yData, y, xData, x, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.atan2(y, x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func cast<S, E, RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  ) where E.Value: BinaryFloatingPoint, RE.Value: BinaryInteger {
    guard useGpu else {
      cpu_cast(from: a, to: &out)
      return
    }
    diagnostic(.queueGpu, "cast(\(a.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      a.withTensor(using: self) { a, aDesc in
        srtCopy(a, aDesc, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.cast(from: a, to: &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func cast<S, E, RE>(
    from a: Tensor<S, E>,
    to out: inout Tensor<S, RE>
  )
  where E.Value: BinaryInteger, RE.Value: BinaryFloatingPoint {
    guard useGpu else {
      cpu_cast(from: a, to: &out)
      return
    }
    diagnostic(.queueGpu, "cast(\(a.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      a.withTensor(using: self) { a, aDesc in
        srtCopy(a, aDesc, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.cast(from: a, to: &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func hypot<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_hypot(x, y, &out)
      return
    }
    diagnostic(.queueGpu, "hypot(\(x.name), \(y.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        y.withTensor(using: self) { yData, y in
          srtHypot(xData, x, yData, y, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.hypot(x, y, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func log<S, E>(
    onePlus x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    guard useGpu else {
      cpu_log(onePlus: x, &out)
      return
    }
    diagnostic(.queueGpu, "log(onePlus: \(x.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        srtLogOnePlus(x, xDesc, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.log(onePlus: x, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func pow<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pow(x, y, &out)
      return
    }
    diagnostic(.queueGpu, "pow(x: \(x.name), y: \(y.name))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        y.withTensor(using: self) { yData, y in
          srtPow(xData, x, yData, y, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.pow(x, y, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func pow<S, E>(
    _ x: Tensor<S, E>,
    _ n: Int,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pow(x, n, &out)
      return
    }
    diagnostic(.queueGpu, "pow(x: \(x.name), n: \(n))", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        withUnsafePointer(to: E.Value(exactly: n)!) { exponent in
          srtPowTE(x, xDesc, exponent, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.pow(x, n, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func root<S, E>(
    _ x: Tensor<S, E>,
    _ n: Int,
    _ out: inout Tensor<S, E>
  ) where E.Value: Real {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_root(x, n, &out)
      return
    }
    diagnostic(.queueGpu, "root(x: \(x.name), n: \(n))", categories: .queueGpu)
    let e = 1 / E.Value(exactly: n)!

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        withUnsafePointer(to: e) { exponent in
          srtPowTE(x, xDesc, exponent, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.root(x, n, &out) }
  }
}
