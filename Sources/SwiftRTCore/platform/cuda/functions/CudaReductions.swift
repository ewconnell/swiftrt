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

import Numerics
import SwiftRTCuda

//==============================================================================
// CudaQueue functions
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable public func abssum<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: SignedNumeric & Comparable {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_abssum(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "abssum(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtAbsSum(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.abssum(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func all<S>(
    _ x: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_all(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "all(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtAll(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.all(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func any<S>(
    _ x: Tensor<S, Bool>,
    _ axis: Int?,
    _ out: inout Tensor<S, Bool>
  ) {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_any(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "any(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtAny(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.any(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func sum<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AdditiveArithmetic {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_sum(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "sum(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtSum(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.sum(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func mean<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: AlgebraicField {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_mean(x, axis, &out)
      return
    }
    // diagnostic(.queueGpu, "mean(\(x.name)) Flat", categories: .queueGpu)

    cpuFallback(cudaErrorNotSupported) { $0.mean(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func min<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_min(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "min(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtMinValue(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.min(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func argmin<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_argmin(x, axis, &arg, &out)
      return
    }
    // diagnostic(.queueGpu, "argmin(\(x.name)) Flat", categories: .queueGpu)

    cpuFallback(cudaErrorNotSupported) { $0.argmin(x, axis, &arg, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func max<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_max(x, axis, &out)
      return
    }

    if out.count == 1 {
      diagnostic(.queueGpu, "max(\(x.name)) Flat", categories: .queueGpu)
      status = out.withMutableTensor(using: self) { o, oDesc in
        x.withTensor(using: self) { x, xDesc in
          srtMaxValue(x, xDesc, o, oDesc, stream)
        }
      }
    } else {
      status = cudaErrorNotSupported
    }
    cpuFallback(status) { $0.max(x, axis, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func argmax<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>
  ) where E.Value: Comparable & ComparableLimits {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_argmax(x, axis, &arg, &out)
      return
    }
    // diagnostic(.queueGpu, "argmax(\(x.name)) Flat", categories: .queueGpu)

    cpuFallback(cudaErrorNotSupported) { $0.argmax(x, axis, &arg, &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable public func prod<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_prod(x, axis, &out)
      return
    }
    // diagnostic(.queueGpu, "prod(\(x.name)) Flat", categories: .queueGpu)

    cpuFallback(cudaErrorNotSupported) { $0.prod(x, axis, &out) }
  }
  //--------------------------------------------------------------------------
  @inlinable public func prodNonZeros<S, E>(
    _ x: Tensor<S, E>,
    _ axis: Int?,
    _ out: inout Tensor<S, E>
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_prodNonZeros(x, axis, &out)
      return
    }
    // diagnostic(.queueGpu, "prodNonZeros(\(x.name)) Flat", categories: .queueGpu)

    cpuFallback(cudaErrorNotSupported) { $0.prodNonZeros(x, axis, &out) }
  }
}
