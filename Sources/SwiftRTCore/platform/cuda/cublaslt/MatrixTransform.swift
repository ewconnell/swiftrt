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
import SwiftRTCuda


//==============================================================================
// MatrixLayout
public final class MatrixTransform: CustomStringConvertible {
    // properties
    public var desc: cublasLtMatrixTransformDesc_t

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(type: srtDataType) {
        let scaleType = cudaDataType(type)
        var temp: cublasLtMatrixTransformDesc_t?
        cudaCheck(cublasLtMatrixTransformDescCreate(&temp, scaleType))
        desc = temp!
        cudaCheck(cublasLtMatrixTransformDescInit(desc, scaleType))
    }

    @inlinable deinit {
        cudaCheck(cublasLtMatrixTransformDescDestroy(desc))
    }

    @inlinable public var description: String {
        """
        MatrixTransform
        elementType: \(elementType)
        pointerMode: \(pointerMode)
        transA: \(transA)
        transB: \(transB)
        """
    }

    //--------------------------------------------------------------------------
    /// getAttribute
    @inlinable public func getAttribute<T>(
        _ attr: cublasLtMatrixTransformDescAttributes_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatrixTransformDescGetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value), &written))
    }

    /// setAttribute
    @inlinable public func setAttribute<T>(
        _ attr: cublasLtMatrixTransformDescAttributes_t,
         _ value: T
    ) {
        var newValue = value
        cudaCheck(cublasLtMatrixTransformDescSetAttribute(
            desc, attr, &newValue, MemoryLayout.size(ofValue: newValue)))
    }

    //--------------------------------------------------------------------------
    /// Specifies the tensor element type
    @inlinable public var elementType: srtDataType {
        get {
            var value = real32F
            getAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, &value)
            return value
        }
        set {
            setAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE, newValue)
        }
    }

    //--------------------------------------------------------------------------
    /// Specifies alpha and beta are passed by reference, whether they
    /// are scalars on the host or on the device, or device vectors. 
    /// Default value is `host`
    @inlinable public var pointerMode: MatmulPointerMode {
        get {
            var value = CUBLASLT_POINTER_MODE_HOST
            getAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, &value)
            return MatmulPointerMode(value)
        }
        set {
            setAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, 
                         newValue.cublas)
        }
    }

    //--------------------------------------------------------------------------
    /// Specifies the type of transformation operation that
    /// should be performed on matrix A. Default value is `noTranspose`
    @inlinable public var transA: TransposeOp {
        get {
            var value = CUBLAS_OP_N
            getAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &value)
            return TransposeOp(value)
        }
        set {
            setAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, newValue.cublas)
        }
    }

    //--------------------------------------------------------------------------
    /// Specifies the type of transformation operation that
    /// should be performed on matrix B. Default value is `noTranspose`
    @inlinable public var transB: TransposeOp {
        get {
            var value = CUBLAS_OP_N
            getAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &value)
            return TransposeOp(value)
        }
        set {
            setAttribute(CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, newValue.cublas)
        }
    }
}