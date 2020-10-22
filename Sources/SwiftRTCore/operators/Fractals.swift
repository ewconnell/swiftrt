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
    tolerance: E.Value,
    range: (first: Complex<E>, last: Complex<E>),
    size: (r: Int, c: Int)
) -> TensorR2<E> where E: Real {
    // generate distributed values over the range
    let iFirst = Complex<E>(0, range.first.imaginary)
    let rFirst = Complex<E>(range.first.real, 0)
    let rLast  = Complex<E>(range.last.real, 0)
    let iLast  = Complex<E>(0, range.last.imaginary)

    // repeat rows of real range, columns of imaginary range, and combine
    let Z = repeating(array(from: rFirst, to: rLast, (1, size.c)), size) +
            repeating(array(from: iFirst, to: iLast, (size.r, 1)), size)
    var d = full(size, iterations, type: E.self)

    currentQueue.juliaSet(Z: Z, constant: C, divergence: d,
                          tolerance: tolerance, 
                          iterations: iterations)
}

//==============================================================================
// CpuQueue delegation
extension CpuQueue {
    @inlinable public func juliaSet<E>(
        Z: TensorR2<Complex<E>>,
        constant C: Complex<E>,
        divergence: inout TensorR2<E>,
        tolerance: E.Value,
        iterations: Int
    ) {
        cpu_juliaSet(Z, C, &divergence, tolerance, iterations)
    }
}

//==============================================================================
// DeviceQueue cpu implementation
extension DeviceQueue {
    @inlinable public func cpu_juliaSet<E>(
        _ Z: TensorR2<Complex<E>>,
        _ C: Complex<E>,
        _ divergence: inout TensorR2<E>,
        _ tolerance: E.Value,
        _ iterations: Int
    ) {
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }
    }
}

//==============================================================================
// CudaQueue gpu implementation
extension CudaQueue {
    @inlinable public func juliaSet<E>(
        Z: TensorR2<Complex<E>>,
        constant C: Complex<E>,
        divergence d: inout TensorR2<E>,
        tolerance: E.Value,
        iterations: Int
    ) {
        assert(Z.isContiguous && d.isContiguous, 
            _messageElementsMustBeContiguous)
        assert(Z.order == d.order, _messageTensorOrderMismatch)

        guard useGpu else { cpu_juliaSet(Z, C, &d, tolerance, iterations); return }
        diagnostic(.queueGpu, "juliaSet(\(Z.name), \(d.name))", 
                    categories: .queueGpu)

        let status = srtJuliaFlat(
            E.type,
            Z.deviceRead(using: self),
            withUnsafePointer(to: C) { $0 },
            d.deviceReadWrite(using: self),
            d.count,
            withUnsafePointer(to: tolerance) { $0 },
            iterations,
            stream)

        cpuFallback(status) { $0.juliaSet(Z, C, &d, tolerance, iterations) }
    }
}
