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

        cpuFallback("add R\(S.rank) \(E.self)",
            // try on gpu
            out.withMutableTensor(using: self) { oData, o in
                lhs.withTensor(using: self) { lData, l in
                    rhs.withTensor(using: self) { rData, r in
                        srtAdd(lData, l, rData, r, oData, o, stream)
                    }
                }
            }
        ) {
            $0.add(lhs, rhs, &out)
        }
    }
}