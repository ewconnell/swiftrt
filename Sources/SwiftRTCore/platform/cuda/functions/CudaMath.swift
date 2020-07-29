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
    ) where E.Value: AdditiveArithmetic {
        cudaCheck(srtAdd(
            CUDA_R_32F, // TODO: this should come from E!!
            lhs.deviceRead(using: self),
            rhs.deviceRead(using: self),
            result.deviceReadWrite(using: self),
            UInt32(lhs.count), stream))
    }

    //--------------------------------------------------------------------------
    @inlinable public func add<S,E>(
        _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
        _ result: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic, E.Stored == Float {
        cudaCheck(srtAdd(
            CUDA_R_32F, // TODO: this should come from E!!
            lhs.deviceRead(using: self),
            rhs.deviceRead(using: self),
            result.deviceReadWrite(using: self),
            UInt32(lhs.count), stream))
    }

    //--------------------------------------------------------------------------
    // https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasLt-api
    // samples: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt
    @inlinable func matmul<E>(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) where E.Value: Numeric {
        guard useGpu else {
            cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
            return 
        }
        
    }
    //--------------------------------------------------------------------------
    @inlinable func matmul<E>(
        _ lhs: TensorR3<E>, _ transposeLhs: Bool,
        _ rhs: TensorR3<E>, _ transposeRhs: Bool,
        _ result: inout TensorR3<E>
    ) where E.Value: Numeric {
        guard useGpu else {
            cpu_matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
            return 
        }

    }
}