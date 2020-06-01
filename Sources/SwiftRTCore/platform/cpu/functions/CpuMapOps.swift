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
                queue.async {
                    out.indices.forEach { out[$0] = op() }
                }
            } else {
                out.indices.forEach { out[$0] = op() }
            }
        }
        
        // queue data transfers and execute
        r.readWrite(using: self)
        if r.isBufferIterable {
            execute(r.mutableBuffer, op)
        } else {
            execute(r.stridedElements, op)
        }
    }

    //==========================================================================
    // range
    @inlinable func mapOp<S,E,C>(
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
                queue.async {
                    zip(out.indices, i0).forEach { out[$0] = $1 }
                }
            } else {
                zip(out.indices, i0).forEach { out[$0] = $1 }
            }
        }
        
        // queue data transfers and execute
        r.readWrite(using: self)
        if r.isBufferIterable {
            execute(elements, r.mutableBuffer)
        } else {
            execute(elements, r.stridedElements)
        }
    }
    
    //==========================================================================
    // inplace
    @inlinable func mapOp<S,E>(
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
                queue.async {
                    out.indices.forEach { out[$0] = op(out[$0]) }
                }
            } else {
                out.indices.forEach { out[$0] = op(out[$0]) }
            }
        }
        
        // queue data transfers and execute
        r.readWrite(using: self)
        if r.isBufferIterable {
            execute(r.mutableBuffer, op)
        } else {
            execute(r.stridedElements, op)
        }
    }
    
    //==========================================================================
    // reduction
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (RE.Value, E.Value) -> RE.Value
    ) {
        // the op
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0, _ out: O,
            _ op: @escaping (O.Element, I0.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                queue.async {
                    zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
                }
            } else {
                zip(out.indices, i0).forEach { i, v in
                    let currentValue = out[i]
                    let sum = op(currentValue, v)
                    out[i] = sum
                }
            }
        }
        
        // queue data transfers and execute
        a.read(using: self)
        r.readWrite(using: self)
        
        // repeat `r`s iterator to match `a` to enable operations along axes
        let rstrides = repeatedStrides(matching: r, to: a.shape)
        let relements = StridedElements(mutating: r, a.shape, rstrides)

        if a.isBufferIterable {
            execute(a.buffer, relements, op)
        } else {
            execute(a.stridedElements, relements, op)
        }
    }
    
    //==========================================================================
    // mapOp 1
    @inlinable func mapOp<S,E,RE>(
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
                queue.async {
                    zip(out.indices, i0).forEach { out[$0] = op($1) }
                }
            } else {
                zip(out.indices, i0).forEach { out[$0] = op($1) }
            }
        }

        // queue data transfers and execute
        a.read(using: self)
        r.readWrite(using: self)

        // if layouts match then iterate through buffer elements,
        // iterate using logical element positions
        if haveSameStorageLayout(a, r) {
            execute(a.buffer, r.mutableBuffer, op)
        } else {
            execute(a.stridedElements, r.stridedElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 2
    // TODO: specialize for + - * / to gain 10% perf boost
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (E.Value, E.Value) -> RE.Value
    ) {
        // the op
        func execute<I0: Collection, I1: Collection, O: MutableCollection>(
            _ i0: I0, _ i1: I1, _ out: O,
            _ op: @escaping (I0.Element, I1.Element) -> O.Element
        ) {
            var out = out
            if mode == .async {
                queue.async {
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
        
        //------------------------------------
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        r.readWrite(using: self)

        // if layouts match then iterate through buffer elements,
        // iterate using logical element positions
        if haveSameStorageLayout(a, b, r) {
            execute(a.buffer, b.buffer, r.mutableBuffer, op)
        } else {
            execute(a.stridedElements, b.stridedElements, r.stridedElements, op)
        }
    }

    //==========================================================================
    // mapOpAdd
    // 20% boost over passed in op
    @inlinable func mapOpAdd<S,E>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        // the op
        func execute<I: Collection, O: MutableCollection>(
            _ i0: I, _ i1: I, _ out: O
        ) where I.Element: AdditiveArithmetic, O.Element == I.Element {
            var out = out
            if mode == .async {
                queue.async {
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
        
        //------------------------------------
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        r.readWrite(using: self)
        
        // if layouts match then iterate through buffer elements,
        // iterate using logical element positions
        if haveSameStorageLayout(a, b, r) {
            execute(a.buffer, b.buffer, r.mutableBuffer)
        } else {
            execute(a.stridedElements, b.stridedElements, r.stridedElements)
        }
    }
    
    //==========================================================================
    // mapOpMul
    // 20% boost over passed in op
    @inlinable func mapOpMul<S,E>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ r: inout Tensor<S,E>
    ) where E.Value: Numeric {
        // the op
        func execute<I: Collection, O: MutableCollection>(
            _ i0: I, _ i1: I, _ out: O
        ) where I.Element: Numeric, O.Element == I.Element {
            var out = out
            if mode == .async {
                queue.async {
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
        
        //------------------------------------
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        r.readWrite(using: self)
        
        // if layouts match then iterate through buffer elements,
        // iterate using logical element positions
        if haveSameStorageLayout(a, b, r) {
            execute(a.buffer, b.buffer, r.mutableBuffer)
        } else {
            execute(a.stridedElements, b.stridedElements, r.stridedElements)
        }
    }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ r: inout Tensor<S,R1>,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> R1.Value
    ) {
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
                queue.async {
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
        
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        c.read(using: self)
        r.readWrite(using: self)
        
        // execute right layout combination
        if haveSameStorageLayout(a, b, c, r) {
            execute(a.buffer, b.buffer, c.buffer, r.mutableBuffer, op)
        } else {
            execute(a.stridedElements, b.stridedElements,
                    c.stridedElements, r.stridedElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1, R2>(
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
                queue.async {
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
        
        //------------------------------------
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        c.read(using: self)
        r1.readWrite(using: self)
        r2.readWrite(using: self)

        // execute right layout combination
        if haveSameStorageLayout(a, b, c, r1, r2) {
            execute(a.buffer, b.buffer, c.buffer,
                    r1.mutableBuffer, r2.mutableBuffer, op)
        } else {
            execute(a.stridedElements, b.stridedElements, c.stridedElements,
                    r1.stridedElements, r2.stridedElements, op)
        }
    }
}
