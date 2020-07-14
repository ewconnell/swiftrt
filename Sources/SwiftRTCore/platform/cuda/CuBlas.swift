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
import CCuda

//==============================================================================
/// CublasLtHandle
/// creates and manages the lifetime of a cublas light handle
public final class CublasLtHandle {
    // properties
    public let handle: cublasLtHandle_t

    /// init
    @inlinable public init() {
        do {
            var temp: cublasLtHandle_t?
            try cudaCheck(status: cublasLtCreate(&temp))
            handle = temp!
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    // deinit
    @inlinable deinit {
        do {
            try cudaCheck(status: cublasLtDestroy(handle))
        } catch {
            Context.currentQueue.writeLog("\(releaseString) \(error)")
        }
    }
}

//==============================================================================
// MatmulDescriptor
public final class MatmulDescriptor {
    // properties
    public let desc: cublasLtMatmulDesc_t

    // initializers
    @inlinable public init(
        computeType: cublasComputeType_t,
        scaleType: cudaDataType_t
    ) {
        do {
            // create the descriptor
            var temp: cublasLtMatmulDesc_t?
            try cudaCheck(status: cublasLtMatmulDescCreate(&temp, computeType,
                                                           scaleType))
            desc = temp!
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cublasLtMatmulDescDestroy(desc))
        } catch {
            Context.currentQueue.writeLog("\(releaseString) \(error)")
        }
    }
}

// //==============================================================================
// // MatmulDescriptor
// public final class MatrixLayout {
//     // properties
//     public let desc: cublasLtMatrixLayout_t

//     // initializers
//     @inlinable public init(
//     ) {
//         do {
//             // create the descriptor
//             var temp: cublasLtMatrixLayout_t?
//             try cudaCheck(status: cublasLtMatrixLayoutCreate(
//                 &Adesc, 
//                 CUDA_R_32F, 
//                 transa == CUBLAS_OP_N ? m : k, 
//                 transa == CUBLAS_OP_N ? k : m, 
//                 lda)
//             desc = temp!
//         } catch {
//             Context.currentQueue.writeLog("\(createString) \(error)")
//             fatalError()
//         }
//     }

//     @inlinable deinit {
//         do {
//             try cudaCheck(status: cublasLtMatrixLayoutDestroy(desc))
//         } catch {
//             Context.currentQueue.writeLog("\(releaseString) \(error)")
//         }
//     }
// }