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
extension DeviceQueue {

    //==========================================================================
    // caller defined generator
    @inlinable func mapOp<S,E>(
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping () -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        var out = out.mutableBuffer
        if mode == .sync {
            out.indices.forEach { out[$0] = op() }
        } else {
            queue.async(group: group) {
                out.indices.forEach { out[$0] = op() }
            }
        }
    }

    //==========================================================================
    // range generator
    @inlinable func mapOp<S,E>(
        from first: E.Value,
        to last: E.Value,
        by step: E.Value,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        var out = out.mutableBuffer
        
        if mode == .sync {
            var io = out.indices.startIndex
            for i in 0..<(out.count - 1) {
                out[io] = first + E.Value(exactly: i)! * step
                io = out.index(after: io)
            }
            out[io] = last
        } else {
            queue.async(group: group) {
                var io = out.indices.startIndex
                for i in 0..<(out.count - 1) {
                    out[io] = first + E.Value(exactly: i)! * step
                    io = out.index(after: io)
                }
                out[io] = last
            }
        }
    }

    //==========================================================================
    // inplace
    @inlinable func mapOp<S,E>(
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value) -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        var out = out.mutableBuffer
        
        if mode == .sync {
            out.indices.forEach { out[$0] = op(out[$0]) }
        } else {
            queue.async(group: group) {
                out.indices.forEach { out[$0] = op(out[$0]) }
            }
        }
    }

    //==========================================================================
    // reduction along axes
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ out: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (RE.Value, E.Value) -> RE.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        // the op
        func execute<I0: Collection, O: MutableCollection>(
            _ i0: I0,
            _ out: O,
            _ op: @escaping (O.Element, I0.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, i0).forEach { out[$0] = op(out[$0], $1) }
                }
            }
        }
        
        // repeat `r`s to match `a`'s shape to enable operations along axes
        let mutableElements = LogicalElements<S,RE>(
            a.count,
            a.shape,
            repeatedStrides(matching: out, to: a.shape),
            out.storage,
            out.storageBase,
            out.order,
            out.spanCount)
        
        mutableElements.prepareForReadWrite()
        
        if a.isContiguous {
            execute(a.buffer, mutableElements, op)
        } else {
            execute(a.elements, mutableElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 1
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ out: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value) -> RE.Value
    ) {
        assert(out.isContiguous)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ a: A,
            _ out: O,
            _ op: @escaping (A.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op($1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op($1) }
                }
            }
        }

        // check order because they will not match for order conversion ops
        if a.order == out.order {
            if a.isContiguous {
                execute(a.buffer, out.mutableBuffer, op)
            } else {
                execute(a.elements, out.mutableBuffer, op)
            }
        } else {
            execute(a.elements, out.mutableElements, op)
        }
    }
    
    //==========================================================================
    // mapOp 2
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ out: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> RE.Value
    ) {
        assert(a.order == b.order && a.order == out.order && out.isContiguous,
               _messageLayoutsMustMatch)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, O: MutableCollection>(
            _ a: A, _ b: B, _ out: O,
            _ op: @escaping (A.Element, B.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, zip(a, b)).forEach {
                    out[$0] = op($1.0, $1.1)
                }
            } else {
                queue.async(group: group) {
                    zip(out.indices, zip(a, b)).forEach {
                        out[$0] = op($1.0, $1.1)
                    }
                }
            }
        }
        
        if a.isContiguous {
            if b.isContiguous {
                execute(a.buffer, b.buffer, out.mutableBuffer, op)
            } else {
                execute(a.buffer, b.elements, out.mutableBuffer, op)
            }
        } else {
            if b.isContiguous {
                execute(a.elements, b.buffer, out.mutableBuffer, op)
            } else {
                execute(a.elements, b.elements, out.mutableBuffer, op)
            }
        }
    }

    //==========================================================================
    // mapOp_Additive
    // This triggers a compiler optimization
    @inlinable func mapOp_Additive<S,E>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) where E.Value: AdditiveArithmetic {
        assert(a.order == b.order && a.order == out.order && out.isContiguous,
               _messageLayoutsMustMatch)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, O: MutableCollection>(
            _ a: A, _ b: B, _ out: O,
            _ op: @escaping (A.Element, B.Element) -> O.Element
        ) where A.Element: AdditiveArithmetic, A.Element == B.Element {
            var out = out
            if mode == .sync {
                zip(out.indices, zip(a, b)).forEach { out[$0] = op($1.0, $1.1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, zip(a, b)).forEach {
                        out[$0] = op($1.0, $1.1)
                    }
                }
            }
        }
        
        if a.isContiguous {
            if b.isContiguous {
                execute(a.buffer, b.buffer, out.mutableBuffer, op)
            } else {
                execute(a.buffer, b.elements, out.mutableBuffer, op)
            }
        } else {
            if b.isContiguous {
                execute(a.elements, b.buffer, out.mutableBuffer, op)
            } else {
                execute(a.elements, b.elements, out.mutableBuffer, op)
            }
        }
    }

    //==========================================================================
    // mapOp tensor element
    @inlinable func mapOp<S,E>(
        _ a: Tensor<S,E>,
        _ element: E.Value,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ a: A, _ elt: A.Element, _ out: O,
            _ op: @escaping (A.Element, A.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op($1, elt) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op($1, elt) }
                }
            }
        }
        
        if a.isContiguous {
            execute(a.buffer, element, out.mutableBuffer, op)
        } else {
            execute(a.elements, element, out.mutableBuffer, op)
        }
    }

    //==========================================================================
    // mapOp element tensor
    @inlinable func mapOp<S,E>(
        _ element: E.Value,
        _ a: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ elt: A.Element, _ a: A, _ out: O,
            _ op: @escaping (A.Element, A.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op(elt, $1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op(elt, $1) }
                }
            }
        }
        
        if a.isContiguous {
            execute(element, a.buffer, out.mutableBuffer, op)
        } else {
            execute(element, a.elements, out.mutableBuffer, op)
        }
    }
    
    //==========================================================================
    // mapOp_Additive
    @inlinable func mapOp_Additive<S,E>(
        _ a: Tensor<S,E>,
        _ element: E.Value,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) where E.Value: AdditiveArithmetic {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ a: A, _ elt: A.Element, _ out: O,
            _ op: @escaping (A.Element, A.Element) -> O.Element
        ) where A.Element: AdditiveArithmetic {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op($1, elt) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op($1, elt) }
                }
            }
        }

        if a.isContiguous {
            execute(a.buffer, element, out.mutableBuffer, op)
        } else {
            execute(a.elements, element, out.mutableBuffer, op)
        }
    }

    //==========================================================================
    // mapOp_Additive
    @inlinable func mapOp_Additive<S,E>(
        _ element: E.Value,
        _ a: Tensor<S,E>,
        _ out: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> E.Value
    ) where E.Value: AdditiveArithmetic {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ elt: A.Element, _ a: A, _ out: O,
            _ op: @escaping (A.Element, A.Element) -> O.Element
        ) where A.Element: AdditiveArithmetic {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op(elt, $1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op(elt, $1) }
                }
            }
        }
        
        if a.isContiguous {
            execute(element, a.buffer, out.mutableBuffer, op)
        } else {
            execute(element, a.elements, out.mutableBuffer, op)
        }
    }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, R1>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ out: inout Tensor<S,R1>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> R1.Value
    ) {
        assert(a.order == b.order && a.order == c.order &&
                a.order == out.order && out.isContiguous)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, C: Collection,
                     O: MutableCollection>(
            _ a: A, _ b: B, _ c: C, _ out: O,
            _ op: @escaping (A.Element, B.Element, C.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, zip(a, zip(b, c))).forEach {
                    out[$0] = op($1.0, $1.1.0, $1.1.1)
                }
            } else {
                queue.async(group: group) {
                    zip(out.indices, zip(a, zip(b, c))).forEach {
                        out[$0] = op($1.0, $1.1.0, $1.1.1)
                    }
                }
            }
        }
        
        let out = out.mutableBuffer
        if a.isContiguous {
            if b.isContiguous {
                if c.isContiguous {
                    execute(a.buffer, b.buffer, c.buffer, out, op)
                } else {
                    execute(a.buffer, b.buffer, c.elements, out, op)
                }
            } else {
                if c.isContiguous {
                    execute(a.buffer, b.elements, c.buffer, out, op)
                } else {
                    execute(a.buffer, b.elements, c.elements, out, op)
                }
            }
        } else {
            if b.isContiguous {
                if c.isContiguous {
                    execute(a.elements, b.buffer, c.buffer, out, op)
                } else {
                    execute(a.elements, b.buffer, c.elements, out, op)
                }
            } else {
                if c.isContiguous {
                    execute(a.elements, b.elements, c.buffer, out, op)
                } else {
                    execute(a.elements, b.elements, c.elements, out, op)
                }
            }
        }
    }
    
    //==========================================================================
    // mapOp 3
    @inlinable func mapOp<S,E0, E1, E2, O1, O2>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ out1: inout Tensor<S,O1>,
        _ out2: inout Tensor<S,O2>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> (O1.Value, O2.Value)
    ) {
        assert(a.isContiguous && b.isContiguous && c.isContiguous &&
                out1.isContiguous && out2.isContiguous)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, C: Collection,
                     O1: MutableCollection, O2: MutableCollection>(
            _ a: A, _ b: B, _ c: C, _ o1: O1, _ o2: O2,
            _ op: @escaping (A.Element, B.Element, C.Element)
                -> (O1.Element, O2.Element)
        ) {
            var o1 = o1, o2 = o2
            if mode == .sync {
                zip(zip(o1.indices, o2.indices), zip(a, zip(b, c))).forEach {
                    let (o1v, o2v) = op($1.0, $1.1.0, $1.1.1)
                    o1[$0.0] = o1v
                    o2[$0.1] = o2v
                }
            } else {
                queue.async(group: group) {
                    zip(zip(o1.indices, o2.indices), zip(a, zip(b, c))).forEach {
                        let (o1v, o2v) = op($1.0, $1.1.0, $1.1.1)
                        o1[$0.0] = o1v
                        o2[$0.1] = o2v
                    }
                }
            }
        }
        
        execute(a.buffer, b.buffer, c.buffer,
                out1.mutableBuffer, out2.mutableBuffer, op)
    }
}
