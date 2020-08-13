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
    @inlinable func abs<S,E>(
        _ x: Tensor<S,E>, 
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable & SignedNumeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_abs(x, &out); return }

        let status = out.withMutableTensor(using: self) { oData, o in
            x.withTensor(using: self) { xData, x in
                srtAbs(xData, x, oData, o, stream)
            }
        }

        cpuFallback(status) { $0.abs(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func add<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_add(lhs, rhs, &out); return }

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtAdd(lData, l, rData, r, oData, o, stream)
                }
            }
        }

        cpuFallback(status) { $0.add(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func div<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_div(lhs, rhs, &out); return }

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtDiv(lData, l, rData, r, oData, o, stream)
                }
            }
        }

        cpuFallback(status) { $0.div(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func mul<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_mul(lhs, rhs, &out); return }

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtMul(lData, l, rData, r, oData, o, stream)
                }
            }
        }

        cpuFallback(status) { $0.mul(lhs, rhs, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func subtract<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)
        guard useGpu else { cpu_subtract(lhs, rhs, &out); return }

        let status = out.withMutableTensor(using: self) { oData, o in
            lhs.withTensor(using: self) { lData, l in
                rhs.withTensor(using: self) { rData, r in
                    srtSub(lData, l, rData, r, oData, o, stream)
                }
            }
        }

        cpuFallback(status) { $0.subtract(lhs, rhs, &out) }
    }
}