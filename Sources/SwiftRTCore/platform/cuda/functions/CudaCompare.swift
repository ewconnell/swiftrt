//******************************************************************************
// Copyright 2019 Google LLC
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
import SwiftRTCuda
import Numerics

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
    //--------------------------------------------------------------------------
    @inlinable public func and<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value == Bool {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_and(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "and() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtAnd(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.and(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable func elementsAlmostEqual<S,E>(
        _ lhs: Tensor<S,E>,
        _ rhs: Tensor<S,E>,
        _ tolerance: E.Value,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: SignedNumeric & Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else {
            cpu_elementsAlmostEqual(lhs, rhs, tolerance, &out)
            return
        }
        diagnostic(.queueGpu, "elementsAlmostEqual() on \(name)", 
                   categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    withUnsafePointer(to: tolerance) { t in
                        srtElementsAlmostEqual(lData, l, rData, r, t, oData, o, stream)
                    }
                }
            }
        }
        cpuFallback(status) { $0.elementsAlmostEqual(lhs, rhs, tolerance, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func equal<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Equatable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_equal(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "equal() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtEqual(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.equal(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func greater<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_greater(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "greater() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtGreater(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.greater(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func greaterOrEqual<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_greaterOrEqual(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "greaterOrEqual() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtGreaterOrEqual(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.greaterOrEqual(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func less<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_less(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "less() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtLess(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.less(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func lessOrEqual<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_lessOrEqual(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "lessOrEqual() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtLessOrEqual(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.lessOrEqual(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func max<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_max(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "max() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtMax(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.max(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func min<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_min(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "min() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtMin(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.min(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func notEqual<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,Bool>
    ) where E.Value: Equatable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_notEqual(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "notEqual() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtNotEqual(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.notEqual(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func or<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value == Bool {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_or(lhs, rhs, &out); return }
        diagnostic(.queueGpu, "or() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtOr(lData, l, rData, r, oData, o, stream)
                }
            }
        }
        cpuFallback(status) { $0.or(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable func replace<S,E>(
        _ x: Tensor<S,E>, 
        _ y: Tensor<S,E>,
        _ condition: Tensor<S,Bool>,
        _ out: inout Tensor<S,E>
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(x.order == y.order && x.order == condition.order,
               _messageTensorOrderMismatch)
        guard useGpu else { cpu_replace(x, y, condition, &out); return }
        diagnostic(.queueGpu, "replace() on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { oData, o in
            x.withTensor(using: self) { xData, x in
                y.withTensor(using: self) { yData, y in
                    condition.withTensor(using: self) { cData, c in
                        srtReplace(xData, x, yData, y, cData, c, oData, o, stream)
                    }
                }
            }
        }
        cpuFallback(status) { $0.replace(x, y, condition, &out) }
    }
}
