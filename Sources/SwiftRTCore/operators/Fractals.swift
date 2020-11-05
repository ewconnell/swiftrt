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

//==============================================================================
/// juliaSet
/// test function for generating the julia set
@inlinable public func juliaSet<E>(
  iterations: Int,
  constant C: Complex<E>,
  tolerance: Float,
  range: (first: Complex<E>, last: Complex<E>),
  size: (r: Int, c: Int)
) -> TensorR2<E>
where
  E: StorageElement & BinaryFloatingPoint, E.Value: BinaryFloatingPoint
{
  // generate distributed values over the range
  let iFirst = Complex<E>(0, range.first.imaginary)
  let rFirst = Complex<E>(range.first.real, 0)
  let rLast = Complex<E>(range.last.real, 0)
  let iLast = Complex<E>(0, range.last.imaginary)

  // repeat rows of real range, columns of imaginary range, and combine
  let Z =
    repeating(array(from: rFirst, to: rLast, shape: (1, size.c)), shape: size)
    + repeating(array(from: iFirst, to: iLast, shape: (size.r, 1)), shape: size)
  var divergence = full(shape: size, E.Value(exactly: iterations)!, type: E.self)

  currentQueue.juliaSet(Z, C, tolerance, iterations, &divergence)
  return divergence
}

//==============================================================================
// CpuQueue delegation
extension CpuQueue {
  @inlinable public func juliaSet<E>(
    _ a: TensorR2<Complex<E>>,
    _ C: Complex<E>,
    _ tolerance: Float,
    _ iterations: Int,
    _ out: inout TensorR2<E>
  ) where E: BinaryFloatingPoint, E.Value: BinaryFloatingPoint {
    cpu_juliaSet(a, C, tolerance, iterations, &out)
  }
}

//==============================================================================
// DeviceQueue cpu implementation
extension DeviceQueue {
  @inlinable public func cpu_juliaSet<E>(
    _ a: TensorR2<Complex<E>>,
    _ C: Complex<E>,
    _ tolerance: Float,
    _ iterations: Int,
    _ out: inout TensorR2<E>
  ) where E: BinaryFloatingPoint, E.Value: BinaryFloatingPoint {
    var Z = a
    let ct = Complex<E>(E(tolerance))

    for i in 0..<iterations {
      Z = multiply(Z, Z, add: C)
      out[abs(Z) .> ct] = min(out, i)
    }
  }
}

//==============================================================================
// CudaQueue gpu implementation

#if canImport(SwiftRTCuda)
  import SwiftRTCuda

  extension CudaQueue {
    @inlinable public func juliaSet<E>(
      _ a: TensorR2<Complex<E>>,
      _ C: Complex<E>,
      _ tolerance: Float,
      _ iterations: Int,
      _ out: inout TensorR2<E>
    ) where E: BinaryFloatingPoint, E.Value: BinaryFloatingPoint {
      assert(
        a.isContiguous && out.isContiguous,
        _messageElementsMustBeContiguous)
      assert(a.order == out.order, _messageTensorOrderMismatch)

      guard useGpu else {
        cpu_juliaSet(a, C, tolerance, iterations, &out)
        return
      }
      diagnostic(.queueGpu, "juliaSet(\(a.name)) Flat", categories: .queueGpu)

      let status = withUnsafePointer(to: C) { pC in
        srtJuliaFlat(
          Complex<E>.type,
          a.deviceRead(using: self),
          pC,
          tolerance,
          iterations,
          out.count,
          out.deviceReadWrite(using: self),
          stream)
      }
      cpuFallback(status) { $0.juliaSet(a, C, tolerance, iterations, &out) }
    }
  }
#endif
