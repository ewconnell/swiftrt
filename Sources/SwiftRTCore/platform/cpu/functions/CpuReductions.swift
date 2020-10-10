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
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        diagnostic(.queueCpu, "all(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out) { $0 && $1 }
        } else {
            // set initial value for blending
            copy(from: x[S.zero, out.shape], to: &out)
            reduceAlongAxes(x, &out) { $0 && $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        diagnostic(.queueCpu, "any(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out) { $0 || $1 }
        } else {
            // set initial value for blending
            copy(from: x[S.zero, out.shape], to: &out)
            reduceAlongAxes(x, &out) { $0 || $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        diagnostic(.queueCpu, "sum(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out, +)
        } else {
            reduceAlongAxes(x, &out, +)
        }
    }
    
    //--------------------------------------------------------------------------
    // this doesn't use `mapReduce` because it has to do a final op on
    // the reduction out inside the async closure
    //
    @inlinable public func cpu_reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        diagnostic(.queueCpu, "mean(\(x.name)) on \(name)", categories: .queueCpu)

        // the reduction count is the product of the reduced dimensions
        var prod = x.count
        if out.count > 1 {
            prod = 1
            for i in 0..<S.rank where out.shape[i] == 1 { prod *= x.shape[i] }
        }
        let scale = 1 / E.Value(exactly: prod)!

        // sum
        if out.count == 1 {
            mapReduce(x, &out, +)
        } else {
            reduceAlongAxes(x, &out, +)
        }
        
        // inplace divide by count
        mapOp(&out) { $0 * scale }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        diagnostic(.queueCpu, "min(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out) { Swift.min($0, $1) }
        } else {
            reduceAlongAxes(x, &out) { Swift.min($0, $1) }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        diagnostic(.queueCpu, "max(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out) { $0 > $1 ? $0 : $1 }
        } else {
            reduceAlongAxes(x, &out) { $0 > $1 ? $0 : $1 }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceProd<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "prod(\(x.name)) on \(name)", categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out, *)
        } else {
            reduceAlongAxes(x, &out, *)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceProdNonZeros<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "prodNonZeros(\(x.name)) on \(name)",
            categories: .queueCpu)
        if out.count == 1 {
            mapReduce(x, &out) { $1 == 0 ? $0 : $0 * $1 }
        } else {
            reduceAlongAxes(x, &out) { $1 == 0 ? $0 : $0 * $1 }
        }
    }

    // //--------------------------------------------------------------------------
    // @inlinable func cpu_reduce<S,E>(
    //     _ opName: String,
    //     _ x: Tensor<S,E>,
    //     _ out: inout Tensor<S,E>,
    //     _ type: ReductionType,
    //     _ opNext: @escaping (E.Value, E.Value) -> E.Value,
    //     _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    // ) {
    //     diagnostic(.queueCpu, "\(opName)(\(x.name)) on \(name)",
    //                categories: .queueCpu)
    //     reduceAlongAxes(x, &out, opNext)
        
    //     if let op = opFinal {
    //         mapOp(&out, op)
    //     }
    // }
}

//==============================================================================
// CpuQueue functions with default cpu delegation
extension CpuQueue {
    //--------------------------------------------------------------------------
    @inlinable public func reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) { cpu_reduceAll(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) { cpu_reduceAny(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic { cpu_reduceSum(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField { cpu_reduceMean(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable { cpu_reduceMin(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable { cpu_reduceMax(x, &out) }
    //--------------------------------------------------------------------------
    @inlinable public func reduceProd<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        cpu_reduceProd(x, &out)
    }
    //--------------------------------------------------------------------------
    @inlinable public func reduceProdNonZeros<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        cpu_reduceProdNonZeros(x, &out)
    }
}

