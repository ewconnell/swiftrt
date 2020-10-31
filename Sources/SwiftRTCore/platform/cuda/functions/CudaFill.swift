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

import SwiftRTCuda

//==============================================================================
// CudaQueue fill functions
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable public func copy<S, E>(
    from x: Tensor<S, E>,
    to out: inout Tensor<S, E>
  ) {
    guard useGpu else {
      cpu_copy(from: x, to: &out)
      return
    }

    diagnostic(.queueGpu, "copy(from: \(x.name), to: \(out.name)) Indexed", categories: .queueGpu)
    let status = out.withMutableTensor(using: self) { o, oDesc in
      x.withTensor(using: self) { xData, x in
        srtCopy(xData, x, o, oDesc, stream)
      }
    }
    cpuFallback(status) { $0.copy(from: x, to: &out) }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E: StorageElement>(
    _ out: inout Tensor<S, E>,
    with element: E.Value
  ) {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(&out, with: element)
      return
    }

    diagnostic(.queueGpu, "fill(\(out.name), with: \(element)) Indexed", categories: .queueGpu)
    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: E.stored(value: element)) {
        srtFill(o, oDesc, $0, stream)
      }
    }
    cpuFallback(status) { $0.fill(&out, with: element) }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E>(
    _ out: inout Tensor<S, E>,
    from first: E.Value,
    to last: E.Value,
    by step: E.Value
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(&out, from: first, to: last, by: step)
      return
    }

    diagnostic(
      .queueGpu,
      "fill(\(out.name), from: \(first), to: \(last), by: \(step)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: E.stored(value: first)) { f in
        withUnsafePointer(to: E.stored(value: last)) { l in
          withUnsafePointer(to: E.stored(value: step)) { s in
            srtFillRange(o, oDesc, f, l, s, stream)
          }
        }
      }
    }
    cpuFallback(status) { $0.fill(&out, from: first, to: last, by: step) }
  }

  //--------------------------------------------------------------------------
  @inlinable func eye<S, E: StorageElement>(
    _ out: inout Tensor<S, E>,
    offset: Int
  ) where E.Value: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_eye(&out, offset: offset)
      return
    }

    diagnostic(.queueGpu, "eye(\(out.name), offset: \(offset)) Indexed", categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      srtEye(o, oDesc, offset, stream)
    }
    cpuFallback(status) { $0.eye(&out, offset: offset) }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E>(
    randomUniform out: inout Tensor<S, E>,
    _ lower: E.Value,
    _ upper: E.Value,
    _ seed: RandomSeed
  ) where E.Value: BinaryFloatingPoint {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(randomUniform: &out, lower, upper, seed)
      return
    }

    // create a 64 bit seed
    let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

    diagnostic(
      .queueGpu,
      "fill(randomUniform: \(out.name), lower: \(lower), upper: \(upper), seed: \(seed)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: E.stored(value: lower)) { l in
        withUnsafePointer(to: E.stored(value: upper)) { u in
          srtFillRandomUniform(o, oDesc, l, u, seed64, stream)
        }
      }
    }
    cpuFallback(status) { $0.fill(randomUniform: &out, lower, upper, seed) }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E>(
    randomNormal out: inout Tensor<S, E>,
    _ mean: E.Value,
    _ std: E.Value,
    _ seed: RandomSeed
  ) where E.Value: BinaryFloatingPoint {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(randomNormal: &out, mean, std, seed)
      return
    }

    // create a 64 bit seed
    let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

    diagnostic(
      .queueGpu, 
      "fill(randomNormal: \(out.name), mean: \(mean), std: \(std), ssed: \(seed)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: E.stored(value: mean)) { m in
        withUnsafePointer(to: E.stored(value: std)) { s in
          srtFillRandomNormal(o, oDesc, m, s, seed64, stream)
        }
      }
    }
    cpuFallback(status) { $0.fill(randomNormal: &out, mean, std, seed) }
  }

  //--------------------------------------------------------------------------
  // case where the mean and stddev are not static scalars,
  // but tensor results from previous ops
  @inlinable func fill<S, E>(
    randomNormal out: inout Tensor<S, E>,
    _ mean: Tensor<S, E>,
    _ std: Tensor<S, E>,
    _ seed: RandomSeed
  ) where E.Value: BinaryFloatingPoint {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(randomNormal: &out, mean, std, seed)
      return
    }

    // create a 64 bit seed
    let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

    diagnostic(
      .queueGpu,
      "fill(randomNormal: \(out.name), mean: \(mean.name), std: \(std.name), seed: \(seed)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      srtFillRandomNormalTensorArgs(
        o, oDesc,
        mean.deviceRead(using: self),
        std.deviceRead(using: self),
        seed64,
        stream)
    }
    cpuFallback(status) { $0.fill(randomNormal: &out, mean, std, seed) }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E>(
    randomTruncatedNormal out: inout Tensor<S, E>,
    _ mean: E.Value,
    _ std: E.Value,
    _ seed: RandomSeed
  ) where E.Value: BinaryFloatingPoint {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(randomTruncatedNormal: &out, mean, std, seed)
      return
    }

    // create a 64 bit seed
    let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

    diagnostic(
      .queueGpu,
      "fill(randomTruncatedNormal: \(out.name), mean: \(mean), std: \(std), seed: \(seed)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      withUnsafePointer(to: E.stored(value: mean)) { m in
        withUnsafePointer(to: E.stored(value: std)) { s in
          srtFillRandomTruncatedNormal(o, oDesc, m, s, seed64, stream)
        }
      }
    }
    cpuFallback(status) {
      $0.fill(randomTruncatedNormal: &out, mean, std, seed)
    }
  }

  //--------------------------------------------------------------------------
  @inlinable func fill<S, E>(
    randomTruncatedNormal out: inout Tensor<S, E>,
    _ mean: Tensor<S, E>,
    _ std: Tensor<S, E>,
    _ seed: RandomSeed
  ) where E.Value: BinaryFloatingPoint {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_fill(randomTruncatedNormal: &out, mean, std, seed)
      return
    }

    // create a 64 bit seed
    let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

    diagnostic(
      .queueGpu,
      "fill(randomTruncatedNormal: \(out.name), "
        + "mean: \(mean.name), std: \(std.name), seed: \(seed)) Indexed",
      categories: .queueGpu)

    let status = out.withMutableTensor(using: self) { o, oDesc in
      srtFillRandomTruncatedNormalTensorArgs(
        o, oDesc,
        mean.deviceRead(using: self),
        std.deviceRead(using: self),
        seed64,
        stream)
    }
    cpuFallback(status) {
      $0.fill(randomTruncatedNormal: &out, mean, std, seed)
    }
  }
}
