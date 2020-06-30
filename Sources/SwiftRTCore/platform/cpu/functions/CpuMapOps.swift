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
// The map operations invoke an `execute` function with suitable storage
// element iterators. The iterators are non reference counted unsafe buffer
// pointers to the elements. This is to avoid ARC copy on write when
// capturing for asynchrnous execution. The async operations are safe,
// because tensor storage lifetime is gauranteed by the queue.
// The `execute` function `out` parameter is mutatated but not `inout` to
// handle async capture requirements.
extension DeviceQueue {

    //==========================================================================
    // generator
    @inlinable func mapOp<S,E>(
        _ opName: String,
        _ r: inout Tensor<S,E>,
        _ op: @escaping () -> E.Value
    ) {
        // the op
        func execute<O: MutableCollection>(
            _ out: O,
            _ op: @escaping () -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    out.indices.forEach { out[$0] = op() }
                }
            } else {
                out.indices.forEach { out[$0] = op() }
            }
        }
        
        // queue data transfers and execute
        if r.isBufferIterable {
            execute(r.mutableBuffer, op)
        } else {
            execute(r.mutableElements, op)
        }
    }

    //==========================================================================
    // range
    @inlinable func mapOp<S,E,C>(
        _ opName: String,
        _ elements: C,
        _ r: inout Tensor<S,E>
    ) where C: Collection, C.Element == E.Value {
        // the op
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0,
            _ out: O
        ) where I0.Element == O.Element {
            var out = out
            if mode == .async {
                queue.async(group: group) {
                    diagnostic("\(queueString) \(opName) on \(name)",
                               categories: .queueFunc)
                    zip(out.indices, i0).forEach { out[$0] = $1 }
                }
            } else {
                zip(out.indices, i0).forEach { out[$0] = $1 }
            }
        }
        
        // queue data transfers and execute
        if r.isBufferIterable {
            execute(elements, r.mutableBuffer)
        } else {
            execute(elements, r.mutableElements)
        }
    }
    
    //==========================================================================
    // inplace
    @inlinable func mapOp<S,E>(
        _ opName: String,
        _ r: inout Tensor<S,E>,
        _ op: @escaping (E.Value) -> E.Value
    ) {
        // the op
        func execute<O: MutableCollection>(
            _ out: O,
            _ op: @escaping (O.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    out.indices.forEach { out[$0] = op(out[$0]) }
                }
            } else {
                out.indices.forEach { out[$0] = op(out[$0]) }
            }
        }
        
        // queue data transfers and execute
        if r.isBufferIterable {
            execute(r.mutableBuffer, op)
        } else {
            execute(r.mutableElements, op)
        }
    }
    
    //==========================================================================
    // reduction
    @inlinable func mapOp<S,E,RE>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (RE.Value, E.Value) -> RE.Value
    ) {
        precondition(a.layout == r.layout)
        // the op
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0, _ out: O,
            _ op: @escaping (O.Element, I0.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
                }
            } else {
                zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
            }
        }
        
        // repeat `r`s to match `a`'s shape to enable operations along axes
        let rMutableElements = LogicalElements<S,RE>(
                a.count,
                a.shape,
                repeatedStrides(matching: r, to: a.shape),
                r.storage,
                r.storageBase,
                r.layout,
                r.stridedSpanCount)
        
        rMutableElements.synchronizeForReadWrite()
        
        if a.isBufferIterable {
            execute(a.buffer, rMutableElements, op)
        } else {
            execute(a.elements, rMutableElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 1
    @inlinable func mapOp<S,E,RE>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (E.Value) -> RE.Value
    ) {
        // the op
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0, _ out: O,
            _ op: @escaping (I0.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, i0).forEach { out[$0] = op($1) }
                }
            } else {
                zip(out.indices, i0).forEach { out[$0] = op($1) }
            }
        }

        // check layouts because they will not match for layout conversion ops
        if a.layout == r.layout {
            if a.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, r.mutableBuffer, op)
                } else {
                    execute(a.buffer, r.mutableElements, op)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, r.mutableBuffer, op)
                } else {
                    execute(a.elements, r.mutableElements, op)
                }
            }
        } else {
            execute(a.elements, r.mutableElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 2
    // TODO: specialize for + - * / to gain 10% perf boost
    @inlinable func mapOp<S,E,RE>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (E.Value, E.Value) -> RE.Value
    ) {
        precondition(a.layout == b.layout && a.layout == r.layout,
                     _messageLayoutsMustMatch)
        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O,
            _ op: @escaping (I0.Element, I1.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, i1)).forEach {
                        out[$0] = op($1.0, $1.1)
                    }
                }
            } else {
                zip(out.indices, zip(i0, i1)).forEach {
                    out[$0] = op($1.0, $1.1)
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, b.buffer, r.mutableBuffer, op)
                } else {
                    execute(a.buffer, b.buffer, r.mutableElements, op)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.buffer, b.elements, r.mutableBuffer, op)
                } else {
                    execute(a.buffer, b.elements, r.mutableElements, op)
                }
            }
        } else {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.elements, b.buffer, r.mutableBuffer, op)
                } else {
                    execute(a.elements, b.buffer, r.mutableElements, op)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, b.elements, r.mutableBuffer, op)
                } else {
                    execute(a.elements, b.elements, r.mutableElements, op)
                }
            }
        }
    }

    //==========================================================================
    // mapOpAdd
    // 20% boost over passed in op
    @inlinable func mapOpAdd<S,E>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        precondition(a.layout == b.layout && a.layout == r.layout,
                     _messageLayoutsMustMatch)

        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O
        ) where I0.Element: AdditiveArithmetic,
                I0.Element == I1.Element, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, i1)).forEach {
                        out[$0] = $1.0 + $1.1
                    }
                }
            } else {
                zip(out.indices, zip(i0, i1)).forEach {
                    out[$0] = $1.0 + $1.1
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.buffer, b.elements, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.elements, r.mutableElements)
                }
            }
        } else {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.elements, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.elements, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, b.elements, r.mutableBuffer)
                } else {
                    execute(a.elements, b.elements, r.mutableElements)
                }
            }
        }
    }

    //==========================================================================
    // mapOpSub
    // 20% boost over passed in op
    @inlinable func mapOpSub<S,E>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        precondition(a.layout == b.layout && a.layout == r.layout)
        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O
        ) where I0.Element: AdditiveArithmetic,
                I0.Element == I1.Element, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, i1)).forEach {
                        out[$0] = $1.0 - $1.1
                    }
                }
            } else {
                zip(out.indices, zip(i0, i1)).forEach {
                    out[$0] = $1.0 - $1.1
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.buffer, b.elements, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.elements, r.mutableElements)
                }
            }
        } else {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.elements, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.elements, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, b.elements, r.mutableBuffer)
                } else {
                    execute(a.elements, b.elements, r.mutableElements)
                }
            }
        }
    }
    
    //==========================================================================
    // mapOpMul
    // 20% boost over passed in op
    @inlinable func mapOpMul<S,E>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Numeric {
        precondition(a.layout == b.layout && a.layout == r.layout)
        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O
        ) where I0.Element: Numeric,
                I0.Element == I1.Element, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, i1)).forEach {
                        out[$0] = $1.0 * $1.1
                    }
                }
            } else {
                zip(out.indices, zip(i0, i1)).forEach {
                    out[$0] = $1.0 * $1.1
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.buffer, b.elements, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.elements, r.mutableElements)
                }
            }
        } else {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.elements, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.elements, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, b.elements, r.mutableBuffer)
                } else {
                    execute(a.elements, b.elements, r.mutableElements)
                }
            }
        }
    }
    
    //==========================================================================
    // mapOpDiv
    // 20% boost over passed in op
    @inlinable func mapOpDiv<S,E>(
        _ opName: String,
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        precondition(a.layout == b.layout && a.layout == r.layout)
        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O
        ) where I0.Element: AlgebraicField,
                I0.Element == I1.Element, O.Element == I0.Element
        {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, i1)).forEach {
                        out[$0] = $1.0 / $1.1
                    }
                }
            } else {
                zip(out.indices, zip(i0, i1)).forEach {
                    out[$0] = $1.0 / $1.1
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.buffer, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.buffer, b.elements, r.mutableBuffer)
                } else {
                    execute(a.buffer, b.elements, r.mutableElements)
                }
            }
        } else {
            if b.isBufferIterable {
                if r.isBufferIterable {
                    execute(a.elements, b.buffer, r.mutableBuffer)
                } else {
                    execute(a.elements, b.buffer, r.mutableElements)
                }
            } else {
                if r.isBufferIterable {
                    execute(a.elements, b.elements, r.mutableBuffer)
                } else {
                    execute(a.elements, b.elements, r.mutableElements)
                }
            }
        }
   }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1>(
        _ opName: String,
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ r: inout Tensor<S,R1>,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> R1.Value
    ) {
        precondition(a.layout == b.layout && a.layout == c.layout &&
                     a.layout == r.layout)
        // the op
        func execute<
            I0: Collection,
            I1: Collection,
            I2: Collection,
            O: MutableCollection
        >(
            _ i0: I0, _ i1: I1, _ i2: I2, _ out: O,
            _ op: @escaping (I0.Element, I1.Element, I2.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(out.indices, zip(i0, zip(i1, i2))).forEach {
                        out[$0] = op($1.0, $1.1.0, $1.1.1)
                    }
                }
            } else {
                zip(out.indices, zip(i0, zip(i1, i2))).forEach {
                    out[$0] = op($1.0, $1.1.0, $1.1.1)
                }
            }
        }
        
        if a.isBufferIterable {
            if b.isBufferIterable {
                if c.isBufferIterable {
                    if r.isBufferIterable {
                        execute(a.buffer, b.buffer, c.buffer, r.mutableBuffer, op)
                    } else {
                        execute(a.buffer, b.buffer, c.buffer, r.mutableElements, op)
                    }
                } else {
                    if r.isBufferIterable {
                        execute(a.buffer, b.buffer, c.elements, r.mutableBuffer, op)
                    } else {
                        execute(a.buffer, b.buffer, c.elements, r.mutableElements, op)
                    }
                }
            } else {
                if c.isBufferIterable {
                    if r.isBufferIterable {
                        execute(a.buffer, b.elements, c.buffer, r.mutableBuffer, op)
                    } else {
                        execute(a.buffer, b.elements, c.buffer, r.mutableElements, op)
                    }
                } else {
                    if r.isBufferIterable {
                        execute(a.buffer, b.elements, c.elements, r.mutableBuffer, op)
                    } else {
                        execute(a.buffer, b.elements, c.elements, r.mutableElements, op)
                    }
                }
            }
        } else {
            if b.isBufferIterable {
                if c.isBufferIterable {
                    if r.isBufferIterable {
                        execute(a.elements, b.buffer, c.buffer, r.mutableBuffer, op)
                    } else {
                        execute(a.elements, b.buffer, c.buffer, r.mutableElements, op)
                    }
                } else {
                    if r.isBufferIterable {
                        execute(a.elements, b.buffer, c.elements, r.mutableBuffer, op)
                    } else {
                        execute(a.elements, b.buffer, c.elements, r.mutableElements, op)
                    }
                }
            } else {
                if c.isBufferIterable {
                    if r.isBufferIterable {
                        execute(a.elements, b.elements, c.buffer, r.mutableBuffer, op)
                    } else {
                        execute(a.elements, b.elements, c.buffer, r.mutableElements, op)
                    }
                } else {
                    if r.isBufferIterable {
                        execute(a.elements, b.elements, c.elements, r.mutableBuffer, op)
                    } else {
                        execute(a.elements, b.elements, c.elements, r.mutableElements, op)
                    }
                }
            }
        }
    }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1, R2>(
        _ opName: String,
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ r1: inout Tensor<S,R1>,
        _ r2: inout Tensor<S,R2>,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> (R1.Value, R2.Value)
    ) {
        //------------------------------------
        // the actual operation. `out` is not `inout` because the operation
        // will be deferred in async mode. This is safe because the operations
        // are synchronized via the queue
        func execute<
            I0: Collection,
            I1: Collection,
            I2: Collection,
            O1: MutableCollection,
            O2: MutableCollection
        >(
            _ i0: I0, _ i1: I1, _ i2: I2, _ o1: O1, _ o2: O2,
            _ op2: @escaping (I0.Element, I1.Element, I2.Element)
                -> (O1.Element, O2.Element)
        ) {
            var o1 = o1, o2 = o2
            if mode == .async {
                diagnostic("\(queueString) \(opName) on \(name)",
                           categories: .queueFunc)
                queue.async(group: group) {
                    zip(zip(o1.indices, o2.indices), zip(i0, zip(i1, i2))).forEach
                    {
                        let (o1v, o2v) = op2($1.0, $1.1.0, $1.1.1)
                        o1[$0.0] = o1v
                        o2[$0.1] = o2v
                    }
                }
            } else {
                zip(zip(o1.indices, o2.indices), zip(i0, zip(i1, i2))).forEach {
                    let (o1v, o2v) = op2($1.0, $1.1.0, $1.1.1)
                    o1[$0.0] = o1v
                    o2[$0.1] = o2v
                }
            }
        }
        
        // execute right layout combination
        assert(a.isBufferIterable && b.isBufferIterable && c.isBufferIterable &&
                r1.isBufferIterable && r2.isBufferIterable)
        execute(a.buffer, b.buffer, c.buffer,
                r1.mutableBuffer, r2.mutableBuffer, op)
    }
}
