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

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
    //--------------------------------------------------------------------------
    @inlinable public func add<S,E>(
        _ lhs: Tensor<S,E>, 
        _ rhs: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where S: TensorShape, E.Value: AdditiveArithmetic {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)

        guard useGpu else { cpu_add(lhs, rhs, &out); return }

        if lhs.isBufferIterable && rhs.isBufferIterable {
            // input tensor must be either dense or repeating a single element
            cudaCheck(
                srtAdd(E.type.cuda,
                       lhs.deviceRead(using: self), lhs.spanCount,
                       rhs.deviceRead(using: self), rhs.spanCount,
                       out.deviceReadWrite(using: self), out.spanCount,
                       stream))

        } else {
            // inputs can be strided to support repeating dimensions
            // complex tiled orders are not supported
            assert(lhs.order == .row || lhs.order == .col &&
                   rhs.order == .row || rhs.order == .col,
                   _messageRepeatingStorageOrderNotSupported)

            lhs.strides.withUnsafeInt32Pointer { l in
                rhs.strides.withUnsafeInt32Pointer { r in
                    out.strides.withUnsafeInt32Pointer { o in
                        cudaCheck(
                            srtAddStrided(
                                E.type.cuda, S.rank,
                                lhs.deviceRead(using: self), l,
                                rhs.deviceRead(using: self), r,
                                out.deviceReadWrite(using: self), o,
                                stream))
                    }
                }
            }
        }
    }
}