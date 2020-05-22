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
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions
{
    //--------------------------------------------------------------------------
    @inlinable public func reduceSumAll<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic { cpu_reduceSumAll(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable func reduce<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E.Value, E.Value) -> E.Value,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) { cpu_reduce(x, &result, opId, opNext, opFinal) }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceSumAll<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        result[result.startIndex] = x.indices.reduce(into: E.Value.zero) {
            $0 += x[$1]
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_reduce<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E.Value, E.Value) -> E.Value,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) {
        // repeat result to match `x`
        // this is unusual because we intentionally are writing to
        // repeated storage for result accumulation
        var repeatedResult = Tensor<S,E>(repeating: result, to: x.shape)
        
        // do the reductions
        mapOp(x, &repeatedResult, opNext)
        
        if let op = opFinal {
            mapOp(&result, op)
        }
    }
}
