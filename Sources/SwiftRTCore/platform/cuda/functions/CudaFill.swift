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

//==============================================================================
// CudaQueue fill functions
extension CudaQueue
{
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E: StorageElement>(
        _ result: inout Tensor<S,E>,
        with element: E.Value
    ) {
        guard useGpu else { cpu_fill(&result, with: element); return }

        usingAppThreadQueue {
            cpu_fill(&result, with: element)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E,B>(
        _ result: inout Tensor<S,E>,
        with range: Range<B>
    ) where E: StorageElement, E.Value: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    {
        guard useGpu else { cpu_fill(&result, with: range); return }

        usingAppThreadQueue {
            cpu_fill(&result, with: range)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ result: inout Tensor<S,E>,
        offset: Int
    ) where E.Value: Numeric {
        guard useGpu else { cpu_eye(&result, offset: offset); return }

        usingAppThreadQueue {
            cpu_eye(&result, offset: offset)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        guard useGpu else {
            cpu_fill(randomUniform: &result, lower, upper, seed); return
        }

        usingAppThreadQueue {
            cpu_fill(randomUniform: &result, lower, upper, seed); return
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        guard useGpu else {
            cpu_fill(randomNormal: &result, mean, standardDeviation, seed)
            return
        }

        usingAppThreadQueue {
            cpu_fill(randomNormal: &result, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        guard useGpu else {
            cpu_fill(randomNormal: &result, mean, standardDeviation, seed)
            return
        }

        usingAppThreadQueue {
            cpu_fill(randomNormal: &result, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed)
            return
        }

        usingAppThreadQueue {
            cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed) 
            return
        }

        usingAppThreadQueue {
            cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed) 
        }
    }
}
