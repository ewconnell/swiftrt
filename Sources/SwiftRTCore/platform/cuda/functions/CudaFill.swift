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
extension CudaQueue
{
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E: StorageElement>(
        _ out: inout Tensor<S,E>,
        with element: E.Value
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_fill(&out, with: element); return }

        cpuFallback(cudaErrorNotSupported) { $0.fill(&out, with: element) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E,B>(
        _ out: inout Tensor<S,E>,
        with range: Range<B>
    ) where E: StorageElement, E.Value: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_fill(&out, with: range); return }

        cpuFallback(cudaErrorNotSupported) { $0.fill(&out, with: range) }
    }

    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ out: inout Tensor<S,E>,
        offset: Int
    ) where E.Value: Numeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_eye(&out, offset: offset); return }

        cpuFallback(cudaErrorNotSupported) { $0.eye(&out, offset: offset) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform out: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomUniform: &out, lower, upper, seed); return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.fill(randomUniform: &out, lower, upper, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomNormal: &out, mean, standardDeviation, seed)
            return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.fill(randomNormal: &out, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomNormal: &out, mean, standardDeviation, seed)
            return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.fill(randomNormal: &out, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &out, mean, standardDeviation, seed)
            return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.fill(randomTruncatedNormal: &out, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &out, mean, standardDeviation, seed) 
            return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.fill(randomTruncatedNormal: &out, mean, standardDeviation, seed) 
        }
    }
}
