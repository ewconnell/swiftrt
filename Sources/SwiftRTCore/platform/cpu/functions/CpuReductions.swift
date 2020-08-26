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

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAll<S>(
        _ x: Tensor<S,Bool>,
        _ r: inout Tensor<S,Bool>
    ) {
        func execute<I0: Collection, O: MutableCollection>(_ i0: I0, _ out: O)
        where I0.Element == Bool, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceAll on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                        $0 = $0 && $1
                    }
                }
            } else {
                out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                    $0 = $0 && $1
                }
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceAny<S>(
        _ x: Tensor<S,Bool>,
        _ r: inout Tensor<S,Bool>
    ) {
        func execute<I0: Collection, O: MutableCollection>(_ i0: I0, _ out: O)
        where I0.Element == Bool, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceAny on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                        $0 = $0 || $1
                    }
                }
            } else {
                out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                    $0 = $0 || $1
                }
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceSum<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic
    {
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0, _ out: O
        ) where I0.Element: AdditiveArithmetic, O.Element == I0.Element {
            var out = out
            let start = out.startIndex
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceSum on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    out[start] = i0.reduce(into: I0.Element.zero) { $0 += $1 }
                }
            } else {
                out[start] = i0.reduce(into: I0.Element.zero) { $0 += $1 }
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMean<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AlgebraicField
    {
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0, _ out: O
        ) where I0.Element: AlgebraicField, O.Element == I0.Element {
            var out = out
            let start = out.startIndex
            let count = I0.Element(exactly: i0.count)!
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceMean on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    let sum = i0.reduce(into: I0.Element.zero) { $0 += $1 }
                    out[start] = sum / count
                }
            } else {
                let sum = i0.reduce(into: I0.Element.zero) { $0 += $1 }
                out[start] = sum / count
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMin<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Comparable {
        
        func execute<I0: Collection, O: MutableCollection>(_ i0: I0, _ out: O)
        where I0.Element: Comparable, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceMin on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                        $0 = Swift.min($0, $1)
                    }
                }
            } else {
                // TODO: report this
                // this is 2X faster than: $0 = $0 <= $1 ? $0 : $1
                out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                    $0 = Swift.min($0, $1)
                }
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
    }
    //--------------------------------------------------------------------------
    @inlinable public func cpu_reduceMax<S,E>(
        _ x: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Comparable {
        
        func execute<I0: Collection, O: MutableCollection>(_ i0: I0, _ out: O)
        where I0.Element: Comparable, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic(.queueCpu, "cpu_reduceMax on \(name)",
                           categories: .queueCpu)
                queue.async(group: group) {
                    out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                        $0 = $0 > $1 ? $0 : $1
                    }
                }
            } else {
                // TODO: report this
                // this is 2X faster than: $0 = Swift.max($0, $1)
                out[out.startIndex] = i0.reduce(into: i0[i0.startIndex]) {
                    $0 = $0 > $1 ? $0 : $1
                }
            }
        }
        
        if x.isBufferIterable {
            execute(x.buffer, r.mutableBuffer)
        } else {
            execute(x.elements, r.mutableBuffer)
        }
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
        mapOp(opName, x, &result, opNext)
        
        if let op = opFinal {
            mapOp(opName, &result, op)
        }
    }
}
