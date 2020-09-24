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
        _ output: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping () -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        var out = output.mutableBuffer

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
        _ output: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<O: MutableCollection>(
            _ out: O
        ) where O.Element == E.Value {
            var out = out
            if mode == .sync {
                var io = out.indices.startIndex
                for i in 0..<(out.count - 1) {
                    out[io] = first + O.Element(exactly: i)! * step
                    io = out.index(after: io)
                }
                out[io] = last
            } else {
                queue.async(group: group) {
                    var io = out.indices.startIndex
                    for i in 0..<(out.count - 1) {
                        out[io] = first + O.Element(exactly: i)! * step
                        io = out.index(after: io)
                    }
                    out[io] = last
                }
            }
        }

        if output.order == .row {
            execute(output.mutableBuffer)
        } else {
            execute(output.mutableElements)
        }
    }

    //==========================================================================
    // inplace
    @inlinable func mapOp<S,E>(
        _ output: inout Tensor<S,E>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value) -> E.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)
        var out = output.mutableBuffer
        
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
    @inlinable func reduceAlongAxes<S,E,RE>(
        _ a: Tensor<S,E>,
        _ output: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (RE.Value, E.Value) -> RE.Value
    ) {
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, O: MutableCollection>(
            _ a: A,
            _ out: O,
            _ op: @escaping (O.Element, A.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op(out[$0], $1) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op(out[$0], $1) }
                }
            }
        }
        
        // repeat `r`s to match `a`'s shape to enable operations along axes
        let mutableElements = LogicalElements<S,RE>(
            a.count,
            a.shape,
            repeatedStrides(matching: output, to: a.shape),
            output.storage,
            output.storageBase,
            output.order,
            output.spanCount)
        
        mutableElements.prepareForReadWrite()
        
        if a.isContiguous {
            execute(a.buffer, mutableElements, op)
        } else {
            execute(a.elements, mutableElements, op)
        }
    }

    //==========================================================================
    // mapOp tensor
    @inlinable public func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ output: inout Tensor<S,RE>,
        _ opName: String,
        _ op: @escaping (E.Value, RE.Value) -> RE.Value
    ) {
        diagnostic(.queueCpu, "\(opName) on \(name)", categories: .queueCpu)
        
        func execute<A: Collection, O: MutableCollection>(
            _ a: A,
            _ out: O,
            _ op: @escaping (A.Element, O.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, a).forEach { out[$0] = op($1, out[$0]) }
            } else {
                queue.async(group: group) {
                    zip(out.indices, a).forEach { out[$0] = op($1, out[$0]) }
                }
            }
        }
        
        // check order because they will not match for order conversion ops
        if a.order == output.order {
            if a.isContiguous {
                if output.isContiguous {
                    execute(a.buffer, output.mutableBuffer, op)
                } else {
                    execute(a.buffer, output.mutableElements, op)
                }
            } else {
                if output.isContiguous {
                    execute(a.elements, output.mutableBuffer, op)
                } else {
                    execute(a.elements, output.mutableElements, op)
                }
            }
        } else {
            execute(a.elements, output.mutableElements, op)
        }
    }
    
    //==========================================================================
    // mapOp tensor
    @inlinable public func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ output: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value) -> RE.Value
    ) {
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
        if a.order == output.order {
            if a.isContiguous {
                if output.isContiguous {
                    execute(a.buffer, output.mutableBuffer, op)
                } else {
                    execute(a.buffer, output.mutableElements, op)
                }
            } else {
                if output.isContiguous {
                    execute(a.elements, output.mutableBuffer, op)
                } else {
                    execute(a.elements, output.mutableElements, op)
                }
            }
        } else {
            execute(a.elements, output.mutableElements, op)
        }
    }

    //==========================================================================
    // mapOp tensor tensor
    @inlinable public func mapOp<S,AE,BE,RE>(
        _ a: Tensor<S,AE>,
        _ b: Tensor<S,BE>,
        _ output: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (AE.Value, BE.Value) -> RE.Value
    ) {
        assert(a.order == b.order && a.order == output.order &&
               output.isContiguous, _messageOrdersMustMatch)
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
        
        let out = output.mutableBuffer
        if a.isContiguous {
            if b.isContiguous {
                execute(a.buffer, b.buffer, out, op)
            } else {
                execute(a.buffer, b.elements, out, op)
            }
        } else {
            if b.isContiguous {
                execute(a.elements, b.buffer, out, op)
            } else {
                execute(a.elements, b.elements, out, op)
            }
        }
    }

    //==========================================================================
    // mapOp tensor tensor Element
    @inlinable func mapOp<S,E,RE>(
        _ a: Tensor<S,E>,
        _ b: Tensor<S,E>,
        _ c: E.Value,
        _ output: inout Tensor<S,RE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value, E.Value) -> RE.Value
    ) {
        assert(a.order == b.order && a.order == output.order &&
               output.isContiguous, _messageOrdersMustMatch)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, O: MutableCollection>(
            _ a: A, _ b: B, _ c: A.Element, _ out: O,
            _ op: @escaping (A.Element, B.Element, A.Element) -> O.Element
        ) {
            var out = out
            if mode == .sync {
                zip(out.indices, zip(a, b)).forEach {
                    out[$0] = op($1.0, $1.1, c)
                }
            } else {
                queue.async(group: group) {
                    zip(out.indices, zip(a, b)).forEach {
                        out[$0] = op($1.0, $1.1, c)
                    }
                }
            }
        }
        
        let out = output.mutableBuffer
        if a.isContiguous {
            if b.isContiguous {
                execute(a.buffer, b.buffer, c, out, op)
            } else {
                execute(a.buffer, b.elements, c, out, op)
            }
        } else {
            if b.isContiguous {
                execute(a.elements, b.buffer, c, out, op)
            } else {
                execute(a.elements, b.elements, c, out, op)
            }
        }
    }

    //==========================================================================
    // mapOp tensor element
    @inlinable func mapOp<S,E,OE>(
        _ a: Tensor<S,E>,
        _ element: E.Value,
        _ output: inout Tensor<S,OE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> OE.Value
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
            execute(a.buffer, element, output.mutableBuffer, op)
        } else {
            execute(a.elements, element, output.mutableBuffer, op)
        }
    }

    //==========================================================================
    // mapOp element tensor
    @inlinable func mapOp<S,E,OE>(
        _ element: E.Value,
        _ a: Tensor<S,E>,
        _ output: inout Tensor<S,OE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E.Value, E.Value) -> OE.Value
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
            execute(element, a.buffer, output.mutableBuffer, op)
        } else {
            execute(element, a.elements, output.mutableBuffer, op)
        }
    }

    //==========================================================================
    // mapOp tensor tensor tensor
    @inlinable func mapOp<S,E0, E1, E2, OE>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ output: inout Tensor<S,OE>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> OE.Value
    ) {
        assert(a.order == b.order && a.order == c.order &&
                a.order == output.order && output.isContiguous)
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
        
        let out = output.mutableBuffer
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
    // mapOp tensor tensor tensor out out
    @inlinable func mapOp<S,E0, E1, E2, O1, O2>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: Tensor<S,E2>,
        _ output1: inout Tensor<S,O1>,
        _ output2: inout Tensor<S,O2>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E0.Value, E1.Value, E2.Value) -> (O1.Value, O2.Value)
    ) {
        assert(a.isContiguous && b.isContiguous && c.isContiguous &&
                output1.isContiguous && output2.isContiguous)
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
                output1.mutableBuffer, output2.mutableBuffer, op)
    }

    //==========================================================================
    // mapOp tensor tensor scalar out out
    @inlinable func mapOp<S,E0, E1, E2, O1, O2>(
        _ a: Tensor<S,E0>,
        _ b: Tensor<S,E1>,
        _ c: E2,
        _ output1: inout Tensor<S,O1>,
        _ output2: inout Tensor<S,O2>,
        _ opName: @autoclosure () -> String,
        _ op: @escaping (E0.Value, E1.Value, E2) -> (O1.Value, O2.Value)
    ) {
        assert(a.isContiguous && b.isContiguous && 
               output1.isContiguous && output2.isContiguous)
        diagnostic(.queueCpu, "\(opName()) on \(name)", categories: .queueCpu)

        func execute<A: Collection, B: Collection, C,
                     O1: MutableCollection, O2: MutableCollection>(
            _ a: A, _ b: B, _ c: C, _ o1: O1, _ o2: O2,
            _ op: @escaping (A.Element, B.Element, C) -> (O1.Element, O2.Element)
        ) {
            var o1 = o1, o2 = o2
            if mode == .sync {
                zip(zip(o1.indices, o2.indices), zip(a, b)).forEach {
                    let (o1v, o2v) = op($1.0, $1.1, c)
                    o1[$0.0] = o1v
                    o2[$0.1] = o2v
                }
            } else {
                queue.async(group: group) {
                    zip(zip(o1.indices, o2.indices), zip(a, b)).forEach {
                        let (o1v, o2v) = op($1.0, $1.1, c)
                        o1[$0.0] = o1v
                        o2[$0.1] = o2v
                    }
                }
            }
        }

        execute(a.buffer, b.buffer, c,
                output1.mutableBuffer, output2.mutableBuffer, op)
    }
}
