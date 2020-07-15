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
// MatmulOperation
public final class MatmulOperation {
    // properties
    public let desc: cublasLtMatmulDesc_t

    // initializers
    @inlinable public init(
        computeType: cublasComputeType_t,
        scaleType: cudaDataType_t
    ) {
        var temp: cublasLtMatmulDesc_t?
        cudaCheck(cublasLtMatmulDescCreate(&temp, computeType, scaleType))
        desc = temp!
    }

    @inlinable deinit {
        cudaCheck(cublasLtMatmulDescDestroy(desc))
    }

    @inlinable public func setAttribute<T>(
        _ attr: cublasLtMatmulDescAttributes_t,
         _ value: inout T
    ) {
        cudaCheck(cublasLtMatmulDescSetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value)))
    }

    @inlinable public func getAttribute<T>(
        _ attr: cublasLtMatmulDescAttributes_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatmulDescGetAttribute(
            desc, attr, &value,MemoryLayout.size(ofValue: value), &written))
    }

    @inlinable public var transA: TransposeOp {
        get {
            var value = CUBLAS_OP_N
            getAttribute(CUBLASLT_MATMUL_DESC_TRANSA, &value)
            return TransposeOp(value)
        }
        set {
            var value = newValue.cublas
            setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, &value)
        }
    }

    @inlinable public var transB: TransposeOp {
        get {
            var value = CUBLAS_OP_N
            getAttribute(CUBLASLT_MATMUL_DESC_TRANSB, &value)
            return TransposeOp(value)
        }
        set {
            var value = newValue.cublas
            setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, &value)
        }
    }

    @inlinable public var transC: TransposeOp {
        get {
            var value = CUBLAS_OP_N
            getAttribute(CUBLASLT_MATMUL_DESC_TRANSC, &value)
            return TransposeOp(value)
        }
        set {
            var value = newValue.cublas
            setAttribute(CUBLASLT_MATMUL_DESC_TRANSC, &value)
        }
    }
}

