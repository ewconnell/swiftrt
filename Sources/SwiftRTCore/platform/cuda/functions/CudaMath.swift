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
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where S: TensorShape, E.Value: AdditiveArithmetic {
        assert(result.isContiguous, _messageElementsMustBeContiguous)
        assert(lhs.order == rhs.order, _messageTensorOrderMismatch)

        guard useGpu else { cpu_add(lhs, rhs, &result); return }

        if lhs.isBufferIterable && rhs.isBufferIterable {
            // input tensor must be either dense or repeating a single element
            cudaCheck(srtAdd(
                E.type.cuda,
                lhs.deviceRead(using: self),
                lhs.stridedSpanCount,
                rhs.deviceRead(using: self),
                rhs.stridedSpanCount,
                result.deviceReadWrite(using: self),
                result.stridedSpanCount,
                stream))
        } else {
            // inputs can be strided to support repeating dimensions
            // complex tiled orders are not supported
            assert(lhs.order == .row || lhs.order == .col &&
                   rhs.order == .row || rhs.order == .col,
                   _messageRepeatingStorageOrderNotSupported)

            lhs.strides.withUnsafeInt32Pointer { lhsStrides in
            rhs.strides.withUnsafeInt32Pointer { rhsStrides in
            result.strides.withUnsafeInt32Pointer { resultStrides in

                cudaCheck(srtAddStrided(
                    E.type.cuda,
                    S.rank,
                    lhs.deviceRead(using: self),
                    lhsStrides,
                    rhs.deviceRead(using: self),
                    rhsStrides,
                    result.deviceReadWrite(using: self),
                    resultStrides,
                    stream))
            }
            }
            }
        }
    }
}