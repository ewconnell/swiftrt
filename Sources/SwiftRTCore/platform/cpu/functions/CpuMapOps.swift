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

extension DeviceQueue {
    //==========================================================================
    // reduction
    @inlinable func mapOp<S,E,RE>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,RE>,
        _ op: @escaping (RE.Value, E.Value) -> RE.Value
    ) {
        //------------------------------------
        // the actual operation. `out` is not `inout` because the operation
        // will be deferred in async mode. This is safe because the operations
        // are synchronized via the queue
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
                zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
            }
        }
        
        // queue data transfers and execute
        x.read(using: self)
        result.readWrite(using: self)
        execute(x.buffer, result.mutableBuffer, op)
    }

    //==========================================================================
    // generator
    @inlinable func mapOp<S,E>(
        _ result: inout Tensor<S,E>,
        _ op: @escaping () -> E.Value
    ) {
        result.readWrite(using: self)
        var r = result.mutableBuffer
        if mode == .async {
            queue.async {
                r.indices.forEach { r[$0] = op() }
            }
        } else {
            r.indices.forEach { r[$0] = op() }
        }
    }

    //==========================================================================
    // range
    @inlinable func mapOp<S,E,C>(
        _ elements: C,
        _ result: inout Tensor<S,E>
    ) where C: Collection, C.Element == E.Value {
        result.readWrite(using: self)
        var r = result.mutableBuffer
        if mode == .async {
            queue.async {
                zip(r.indices, elements).forEach { r[$0] = $1 }
            }
        } else {
            zip(r.indices, elements).forEach { r[$0] = $1 }
        }
    }
    
    //==========================================================================
    // inplace
    @inlinable func mapOp<S,E>(
        _ result: inout Tensor<S,E>,
        _ op: @escaping (E.Value) -> E.Value
    ) {
        result.readWrite(using: self)
        var r = result.mutableBuffer
        if mode == .async {
            queue.async {
                r.indices.forEach { r[$0] = op(r[$0]) }
            }
        } else {
            r.indices.forEach { r[$0] = op(r[$0]) }
        }
    }
    
    //==========================================================================
    // mapOp 1
    @inlinable func mapOp<S,E,RE>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,RE>,
        _ op: @escaping (E.Value) -> RE.Value
    ) {
        //------------------------------------
        // the actual operation. `out` is not `inout` because the operation
        // will be deferred in async mode. This is safe because the operations
        // are synchronized via the queue
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
        x.read(using: self)
        result.readWrite(using: self)
        execute(x.buffer, result.mutableBuffer, op)
    }
    
    //==========================================================================
    // mapOp 2
    @inlinable func mapOp<S,E,RE>(
        _ lhs: Tensor<S,E>,
        _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,RE>,
        _ op: @escaping (E.Value, E.Value) -> RE.Value
    ) {
        //------------------------------------
        // the actual operation. `out` is not `inout` because the operation
        // will be deferred in async mode. This is safe because the operations
        // are synchronized via the queue
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
        lhs.read(using: self)
        rhs.read(using: self)
        result.readWrite(using: self)

        // execute right layout combination
        if haveSameStorageLayout(lhs, rhs) {
            execute(lhs.buffer, rhs.buffer, result.mutableBuffer, op)
        } else {
            switch (lhs.layout, rhs.layout) {
            default:
                execute(lhs.stridedElements,
                        lhs.stridedElements,
                        result.stridedElements, op)
            }
        }
    }

    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ result: inout Tensor<S,R1>,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> R1.Value
    ) {
        //------------------------------------
        // the actual operation. `out` is not `inout` because the operation
        // will be deferred in async mode. This is safe because the operations
        // are synchronized via the queue
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
        
        //------------------------------------
        // queue data transfers
        a.read(using: self)
        b.read(using: self)
        c.read(using: self)
        result.readWrite(using: self)
        
        // execute right layout combination
        if haveSameStorageLayout(a, b, c) {
            execute(a.buffer, b.buffer, c.buffer, result.mutableBuffer, op)
        } else {
            switch (a.layout, b.layout, c.layout) {
            default:
                fatalError("mixed layout not implemented")
            }
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
        if haveSameStorageLayout(a, b, c) {
            execute(a.buffer, b.buffer, c.buffer,
                    r1.mutableBuffer, r2.mutableBuffer, op)
        } else {
            switch (a.layout, b.layout, c.layout) {
            default:
                fatalError("mixed layout not implemented")
            }
        }
    }
}
