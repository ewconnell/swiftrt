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
    @inlinable public func mapReduce<S,E>(
        _ a: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        let a = a.buffer
        var out = out.mutableBuffer
        
        if mode == .sync {
            out[out.startIndex] = a.reduce(into: a[a.startIndex]) {
                $0 = op($0, $1)
            }
        } else {
            queue.async(group: group) {
                out[out.startIndex] = a.reduce(into: a[a.startIndex]) {
                    $0 = op($0, $1)
                }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        mapReduce(x, &out, "all(\(x.name))") { $0 && $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        mapReduce(x, &out, "any(\(x.name))") { $0 || $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        mapReduce(x, &out, "sum(\(x.name))") { $0 + $1 }
    }
    
    //--------------------------------------------------------------------------
    // this doesn't use `mapReduce` because it has to do a final op on
    // the reduction result inside the async closure
    //
    @inlinable public func cpu_reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        let a = x.buffer
        var out = out.mutableBuffer
        let count = E.Value(exactly: x.count)!
        let start = out.startIndex
        
        if mode == .sync {
            let sum = a.reduce(into: a[a.startIndex]) { $0 += $1 }
            out[start] = sum / count
        } else {
            diagnostic(.queueCpu, "mean(\(x.name)) on \(name)",
                       categories: .queueCpu)
            queue.async(group: group) {
                let sum = a.reduce(into: a[a.startIndex]) { $0 += $1 }
                out[start] = sum / count
            }
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        mapReduce(x, &out, "min(\(x.name))") { Swift.min($0, $1) }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        mapReduce(x, &out, "max(\(x.name))") { $0 > $1 ? $0 : $1 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_reduce<S,E>(
        _ opName: String,
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E.Value, E.Value) -> E.Value,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) {
        mapOp(x, &result, opName, opNext)
        
        if let op = opFinal {
            mapOp(&result, opName, op)
        }
    }
}

//==============================================================================
// CpuQueue functions with default cpu delegation
extension CpuQueue
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
        _ opName: String,
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E.Value, E.Value) -> E.Value,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) { cpu_reduce(opName, x, &result, opId, opNext, opFinal) }
}

