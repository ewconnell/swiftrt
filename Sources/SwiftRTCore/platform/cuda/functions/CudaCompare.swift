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
  @inlinable public func and<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value == Bool {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_and(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "and(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtAnd(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.and(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func elementsAlmostEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ tolerance: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: SignedNumeric & Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_elementsAlmostEqual(lhs, rhs, tolerance, &out)
      return
    }

    diagnostic(
      .queueGpu, "elementsAlmostEqual(\(lhs.name), \(rhs.name), tolerance: \(tolerance)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          withUnsafePointer(to: tolerance) { t in
            srtElementsAlmostEqual(l, lDesc, r, rDesc, t, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.elementsAlmostEqual(lhs, rhs, tolerance, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func equal<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Equatable {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_equal(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, rhs, out) {
      diagnostic(.queueGpu, "equal(\(lhs.name), \(rhs.name)) Flat", categories: .queueGpu)
      status = srtEqualFlat(
        E.type,
        lhs.deviceRead(using: self),
        rhs.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )
    } else {
      diagnostic(.queueGpu, "equal(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            srtEqual(l, lDesc, r, rDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.equal(lhs, rhs, &out) }
  }

  // greater tensor Element
  @inlinable public func equal<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Equatable {
    var element = rhs
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_equal(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, out) {
      diagnostic(.queueGpu, "equal(\(lhs.name), \(rhs)) Flat", categories: .queueGpu)
      status = srtEqualFlatTE(
          E.type,
          lhs.deviceRead(using: self),
          &element,
          out.deviceReadWrite(using: self),
          out.count,
          stream
        )

    } else {
      diagnostic(.queueGpu, "equal(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          srtEqualTE(l, lDesc, &element, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.equal(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // greater

  // greater tensor tensor
  @inlinable public func greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_greater(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, rhs, out) {
      diagnostic(.queueGpu, "greater(\(lhs.name), \(rhs.name)) Flat", categories: .queueGpu)
      status = srtGreaterFlat(
        E.type,
        lhs.deviceRead(using: self),
        rhs.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )
    } else {
      diagnostic(.queueGpu, "greater(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            srtGreater(l, lDesc, r, rDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.greater(lhs, rhs, &out) }
  }

  // greater tensor Element
  @inlinable public func greater<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    var element = rhs
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_greater(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, out) {
      diagnostic(.queueGpu, "greater(\(lhs.name), \(rhs)) Flat", categories: .queueGpu)
      status = srtGreaterFlatTE(
          E.type,
          lhs.deviceRead(using: self),
          &element,
          out.deviceReadWrite(using: self),
          out.count,
          stream
        )

    } else {
      diagnostic(.queueGpu, "greater(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          srtGreaterTE(l, lDesc, &element, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.greater(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // greaterOrEqual

  // greaterOrEqual tensor tensor
  @inlinable public func greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_greaterOrEqual(lhs, rhs, &out)
      return
    }

    diagnostic(
      .queueGpu, "greaterOrEqual(\(lhs.name), \(rhs.name)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtGreaterOrEqual(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.greaterOrEqual(lhs, rhs, &out) }
  }

  // greaterOrEqual tensor Element
  @inlinable public func greaterOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_greaterOrEqual(lhs, rhs, &out)
      return
    }

    diagnostic(
      .queueGpu, "greaterOrEqual(\(lhs.name), \(rhs)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtGreaterOrEqualTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.greaterOrEqual(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // less

  // less tensor tensor
  @inlinable public func less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_less(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "less(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtLess(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.less(lhs, rhs, &out) }
  }

  // less tensor Element
  @inlinable public func less<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_less(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "less(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtLessTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.less(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // lessOrEqual

  // lessOrEqual tensor tensor
  @inlinable public func lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_lessOrEqual(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "lessOrEqual(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtLessOrEqual(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.lessOrEqual(lhs, rhs, &out) }
  }

  // less tensor Element
  @inlinable public func lessOrEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_lessOrEqual(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "lessOrEqual(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtLessOrEqualTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.lessOrEqual(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  // min

  // min tensor tensor
  @inlinable public func min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_min(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, rhs, out) {
      diagnostic(.queueGpu, "min(\(lhs.name), \(rhs.name)) Flat", categories: .queueGpu)
      status = srtMinFlat(
        E.type,
        lhs.deviceRead(using: self),
        rhs.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )
    } else {
      diagnostic(.queueGpu, "min(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          rhs.withTensor(using: self) { r, rDesc in
            srtMin(l, lDesc, r, rDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.min(lhs, rhs, &out) }
  }

  // min tensor Element
  @inlinable public func min<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_min(lhs, rhs, &out)
      return
    }

    if canFlatten(lhs, out) {
      diagnostic(.queueGpu, "min(\(lhs.name), \(rhs)) Flat", categories: .queueGpu)
      status = withUnsafePointer(to: rhs) { prhs in
        srtMinFlatTE(
          E.type,
          lhs.deviceRead(using: self),
          prhs,
          out.deviceReadWrite(using: self),
          out.count,
          stream
        )
      }
    } else {
      diagnostic(.queueGpu, "min(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        lhs.withTensor(using: self) { l, lDesc in
          withUnsafePointer(to: rhs) { r in
            srtMinTE(l, lDesc, r, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.min(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_max(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "max(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtMax(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.max(lhs, rhs, &out) }
  }

  // max tensor Element
  @inlinable public func max<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: E.Value,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_max(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "max(\(lhs.name), \(rhs)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        withUnsafePointer(to: rhs) { r in
          srtMaxTE(l, lDesc, r, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.max(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func notEqual<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, Bool>
  ) where E.Value: Equatable {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_notEqual(lhs, rhs, &out)
      return
    }
    diagnostic(.queueGpu, "notEqual(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtNotEqual(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.notEqual(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func or<S, E>(
    _ lhs: Tensor<S, E>,
    _ rhs: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value == Bool {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_or(lhs, rhs, &out)
      return
    }

    diagnostic(.queueGpu, "or(\(lhs.name), \(rhs.name)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      lhs.withTensor(using: self) { l, lDesc in
        rhs.withTensor(using: self) { r, rDesc in
          srtOr(l, lDesc, r, rDesc, o, oDesc, stream)
        }
      }
    }
    cpuFallback(status) { $0.or(lhs, rhs, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func replace<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ condition: Tensor<S, Bool>,
    _ out: inout Tensor<S, E>
  ) {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    assert(
      x.order == y.order && x.order == condition.order,
      _messageTensorOrderMismatch)
    guard useGpu else {
      cpu_replace(x, y, condition, &out)
      return
    }

    if canFlatten(x, y, condition, out) {
      diagnostic(
        .queueGpu,
        "replace(x: \(x.name), y: \(y.name), condition: \(condition.name)) Flat",
        categories: .queueGpu)
        
      status = srtReplaceFlat(
        E.type,
        x.deviceRead(using: self),
        y.deviceRead(using: self),
        boolean,
        condition.deviceRead(using: self),
        out.deviceReadWrite(using: self),
        out.count,
        stream
      )
    } else {
      diagnostic(
        .queueGpu,
        "replace(x: \(x.name), y: \(y.name), condition: \(condition.name)) Indexed",
        categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          y.withTensor(using: self) { y, yDesc in
            condition.withTensor(using: self) { c, cDesc in
              srtReplace(x, xDesc, y, yDesc, c, cDesc, o, oDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.replace(x, y, condition, &out) }
  }

  //==========================================================================
  @inlinable func vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMin(x, y, scale, &out)
      return
    }

    diagnostic(
      .queueGpu, "vjpMin(\(x.name), \(y.name), \(scale.name)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        y.withTensor(using: self) { y, yDesc in
          scale.withTensor(using: self) { s, sDesc in
            srtVjpMin(x, xDesc, y, yDesc, s, sDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMin(x, y, scale, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMin(x, y, scale, &out)
      return
    }

    diagnostic(
      .queueGpu, "vjpMin(\(x.name), \(y), \(scale.name)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        withUnsafePointer(to: y) { y in
          scale.withTensor(using: self) { s, sDesc in
            srtVjpMinTE(x, xDesc, y, s, sDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMin(x, y, scale, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ outT: inout Tensor<S, E>,
    _ outF: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(
      outT.isContiguous && outF.isContiguous,
      _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMin(x, y, scale, &outT, &outF)
      return
    }

    diagnostic(
      .queueGpu, "vjpMin(\(x.name), \(y.name), \(scale.name)) Indexed",
      categories: .queueGpu)
    var outFShared = outF.shared(using: self)

    let status = outT.withMutableTensor(using: self) { oT, oTDesc in
      outFShared.withMutableTensor(using: self) { oF, oFDesc in
        x.withTensor(using: self) { x, xDesc in
          y.withTensor(using: self) { y, yDesc in
            scale.withTensor(using: self) { s, sDesc in
              srtVjpMinOO(
                x, xDesc, y, yDesc, s, sDesc,
                oT, oTDesc, oF, oFDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMin(x, y, scale, &outT, &outF) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMin<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ outT: inout Tensor<S, E>,
    _ outF: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(
      outT.isContiguous && outF.isContiguous,
      _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMin(x, y, scale, &outT, &outF)
      return
    }

    diagnostic(
      .queueGpu, "vjpMin(\(x.name), \(y), \(scale.name)) Indexed",
      categories: .queueGpu)
    var outFShared = outF.shared(using: self)

    let status = outT.withMutableTensor(using: self) { oT, oTDesc in
      outFShared.withMutableTensor(using: self) { oF, oFDesc in
        x.withTensor(using: self) { x, xDesc in
          withUnsafePointer(to: y) { y in
            scale.withTensor(using: self) { s, sDesc in
              srtVjpMinTEOO(
                x, xDesc, y, s, sDesc,
                oT, oTDesc, oF, oFDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMin(x, y, scale, &outT, &outF) }
  }

  //==========================================================================
  @inlinable func vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMax(x, y, scale, &out)
      return
    }

    diagnostic(
      .queueGpu, "vjpMax(\(x.name), \(y.name), \(scale.name)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        y.withTensor(using: self) { y, yDesc in
          scale.withTensor(using: self) { s, sDesc in
            srtVjpMax(x, xDesc, y, yDesc, s, sDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMax(x, y, scale, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMax(x, y, scale, &out)
      return
    }
    diagnostic(
      .queueGpu, "vjpMax(\(x.name), \(y), \(scale.name)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { x, xDesc in
        withUnsafePointer(to: y) { y in
          scale.withTensor(using: self) { s, sDesc in
            srtVjpMaxTE(x, xDesc, y, s, sDesc, o, oDesc, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMax(x, y, scale, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: Tensor<S, E>,
    _ scale: Tensor<S, E>,
    _ outT: inout Tensor<S, E>,
    _ outF: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(
      outT.isContiguous && outF.isContiguous,
      _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMax(x, y, scale, &outT, &outF)
      return
    }

    diagnostic(
      .queueGpu, "vjpMax(\(x.name), \(y.name), \(scale.name)) Indexed",
      categories: .queueGpu)
    var outFShared = outF.shared(using: self)

    let status = outT.withMutableTensor(using: self) { oT, oTDesc in
      outFShared.withMutableTensor(using: self) { oF, oFDesc in
        x.withTensor(using: self) { x, xDesc in
          y.withTensor(using: self) { y, yDesc in
            scale.withTensor(using: self) { s, sDesc in
              srtVjpMaxOO(
                x, xDesc, y, yDesc, s, sDesc,
                oT, oTDesc, oF, oFDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMax(x, y, scale, &outT, &outF) }
  }

  //--------------------------------------------------------------------------
  @inlinable func vjpMax<S, E>(
    _ x: Tensor<S, E>,
    _ y: E.Value,
    _ scale: Tensor<S, E>,
    _ outT: inout Tensor<S, E>,
    _ outF: inout Tensor<S, E>
  ) where E.Value: Comparable & Numeric {
    assert(
      outT.isContiguous && outF.isContiguous,
      _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_vjpMax(x, y, scale, &outT, &outF)
      return
    }

    diagnostic(
      .queueGpu, "vjpMax(\(x.name), \(y), \(scale.name)) Indexed",
      categories: .queueGpu)
    var outFShared = outF.shared(using: self)

    let status = outT.withMutableTensor(using: self) { oT, oTDesc in
      outFShared.withMutableTensor(using: self) { oF, oFDesc in
        x.withTensor(using: self) { x, xDesc in
          withUnsafePointer(to: y) { y in
            scale.withTensor(using: self) { s, sDesc in
              srtVjpMaxTEOO(
                x, xDesc, y, s, sDesc,
                oT, oTDesc, oF, oFDesc, stream)
            }
          }
        }
      }
    }
    cpuFallback(status) { $0.vjpMax(x, y, scale, &outT, &outF) }
  }
}
