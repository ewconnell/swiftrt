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
import Numerics

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions
{
    //--------------------------------------------------------------------------
    @inlinable public func reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ result: inout Tensor<S,Bool>
    ) { cpu_reduceAll(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ result: inout Tensor<S,Bool>
    ) { cpu_reduceAny(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic { cpu_reduceSum(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: AlgebraicField { cpu_reduceMean(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: Comparable { cpu_reduceMin(x, &result) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: Comparable { cpu_reduceMax(x, &result) }
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
    @inlinable public func cpu_reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ r: inout Tensor<S,Bool>
    ) {
        r[r.startIndex] = x.buffer.reduce(into: x[x.startIndex]) { $0 = $0 && $1 }
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ r: inout Tensor<S,Bool>
    ) {
        r[r.startIndex] = x.buffer.reduce(into: x[x.startIndex]) { $0 = $0 || $1 }
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        r[r.startIndex] = x.buffer.reduce(into: E.Value.zero) { $0 += $1 }
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        let sum = x.buffer.reduce(into: E.Value.zero) { $0 += $1 }
        r[r.startIndex] = sum / E.Value(exactly: x.count)!
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Comparable {
        r[r.startIndex] = x.buffer.reduce(into: x[x.startIndex]) {
            // this is 2X faster than: $0 = $0 <= $1 ? $0 : $1
            $0 = Swift.min($0, $1)
        }
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Comparable {
        r[r.startIndex] = x.buffer.reduce(into: x[x.startIndex]) {
            // this is 2X faster than: $0 = Swift.max($0, $1)
            $0 = $0 > $1 ? $0 : $1
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
