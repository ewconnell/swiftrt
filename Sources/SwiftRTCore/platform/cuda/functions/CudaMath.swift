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

        let lData = lhs.deviceRead(using: self)
        let rData = rhs.deviceRead(using: self)
        let oData = out.deviceReadWrite(using: self)

        lhs.withTensorDescriptor { l in
            rhs.withTensorDescriptor { r in
                out.withTensorDescriptor { o in
                    // compile time switch for static binding
                    switch (S.rank, E.self) {
                    case (1, is Float.Type): srtAddR1Float(lData, l, rData, r, oData, o, stream)
                    case (2, is Float.Type): srtAddR2Float(lData, l, rData, r, oData, o, stream)
                    case (3, is Float.Type): srtAddR3Float(lData, l, rData, r, oData, o, stream)

                    case (1, is Float16.Type): srtAddR1Float16(lData, l, rData, r, oData, o, stream)
                    case (2, is Float16.Type): srtAddR2Float16(lData, l, rData, r, oData, o, stream)
                    case (3, is Float16.Type): srtAddR3Float16(lData, l, rData, r, oData, o, stream)

                    default:
                        diagnostic("\(fallbackString) add R\(S.rank) \(E.self)",
                                   categories: .fallback) 
                        Context.appThreadQueue.add(lhs, rhs, &out)
                    }
                }
            }
        }
        cudaCheck(stream)
    }
}