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
import Numerics

//==============================================================================
// CudaQueue functions
extension CudaQueue {
    //--------------------------------------------------------------------------
    @inlinable public func reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceAll(x, &out); return }
        diagnostic(.queueGpu, "reduceAll() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceAll(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceAny(x, &out); return }
        diagnostic(.queueGpu, "reduceAny() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceAny(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceSum(x, &out); return }
        diagnostic(.queueGpu, "reduceSum() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceSum(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceMean(x, &out); return }
        diagnostic(.queueGpu, "reduceMean() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceMean(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceMin(x, &out); return }
        diagnostic(.queueGpu, "reduceMin() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceMin(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_reduceMax(x, &out); return }
        diagnostic(.queueGpu, "reduceMax() on \(name)", categories: .queueGpu)
        
        cpuFallback(cudaErrorNotSupported) { $0.reduceMax(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable func reduce<S,E>(
        _ opName: String,
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E.Value, E.Value) -> E.Value,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_reduce(opName, x, &out, opId, opNext, opFinal); return
        }

        cpuFallback(cudaErrorNotSupported) {
            $0.reduce(opName, x, &out, opId, opNext, opFinal)
        }
    }
}
