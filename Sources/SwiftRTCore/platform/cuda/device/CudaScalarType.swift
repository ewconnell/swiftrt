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

//------------------------------------------------------------------------------
// StorageElementType extension
extension cudaDataType_t : Hashable {}

// cuda data types conversion
extension StorageElementType {
    @inlinable public init(_ cudaType: cudaDataType_t) {
        let types: [cudaDataType_t: StorageElementType] = [
            CUDA_R_16F: .real16F,
            CUDA_C_16F: .complex16F,
            CUDA_R_16BF: .real16BF,
            CUDA_C_16BF: .complex16BF,
            CUDA_R_32F: .real32F,
            CUDA_C_32F: .complex32F,
            CUDA_R_64F: .real64F,
            CUDA_C_64F: .complex64F,
            CUDA_R_4I: .real4I,
            CUDA_C_4I: .complex4I,
            CUDA_R_4U: .real4U,
            CUDA_C_4U: .complex4U,
            CUDA_R_8I: .real8I,
            CUDA_C_8I: .complex8I,
            CUDA_R_8U: .real8U,
            CUDA_C_8U: .complex8U,
            CUDA_R_16I: .real16I,
            CUDA_C_16I: .complex16I,
            CUDA_R_16U: .real16U,
            CUDA_C_16U: .complex16U,
            CUDA_R_32I: .real32I,
            CUDA_C_32I: .complex32I,
            CUDA_R_32U: .real32U,
            CUDA_C_32U: .complex32U,
            CUDA_R_64I: .real64I,
            CUDA_C_64I: .complex64I,
            CUDA_R_64U: .real64U,
            CUDA_C_64U: .complex64U  
        ]
        assert(types[cudaType] != nil, "Unknown cudaDataType_t")
        self = types[cudaType]!
    }

    @inlinable public var cuda: cudaDataType_t {
        let types: [StorageElementType : cudaDataType_t] = [
            .real16F: CUDA_R_16F,
            .complex16F: CUDA_C_16F,
            .real16BF: CUDA_R_16BF,
            .complex16BF: CUDA_C_16BF,
            .real32F: CUDA_R_32F,
            .complex32F: CUDA_C_32F,
            .real64F: CUDA_R_64F,
            .complex64F: CUDA_C_64F,
            .real4I: CUDA_R_4I,
            .complex4I: CUDA_C_4I,
            .real4U: CUDA_R_4U,
            .complex4U: CUDA_C_4U,
            .real8I: CUDA_R_8I,
            .complex8I: CUDA_C_8I,
            .real8U: CUDA_R_8U,
            .complex8U: CUDA_C_8U,
            .real16I: CUDA_R_16I,
            .complex16I: CUDA_C_16I,
            .real16U: CUDA_R_16U,
            .complex16U: CUDA_C_16U,
            .real32I: CUDA_R_32I,
            .complex32I: CUDA_C_32I,
            .real32U: CUDA_R_32U,
            .complex32U: CUDA_C_32U,
            .real64I: CUDA_R_64I,
            .complex64I: CUDA_C_64I,
            .real64U: CUDA_R_64U,
            .complex64U: CUDA_C_64U  
        ]
        assert(types[self] != nil, "Unknown cudaDataType_t")
        return types[self]!
    }
}

//------------------------------------------------------------------------------
// cudnn data types conversion
extension cudnnDataType_t : Hashable {}

extension StorageElementType {
    @inlinable public init(_ cudnnType: cudnnDataType_t) {
        let types: [cudnnDataType_t : StorageElementType] = [
            CUDNN_DATA_HALF: .real16F,
            CUDNN_DATA_FLOAT: .real32F,
            CUDNN_DATA_DOUBLE: .real64F,
            CUDNN_DATA_INT8: .real8I,
            CUDNN_DATA_INT32: .real32I,
            CUDNN_DATA_UINT8: .real8U,
        ]
        assert(types[cudnnType] != nil, "Unknown cudnnDataType_t")
        self = types[cudnnType]!
    }

    @inlinable public var cudnn: cudnnDataType_t {
        let types: [StorageElementType : cudnnDataType_t] = [
            .real16F: CUDNN_DATA_HALF,
            .real32F: CUDNN_DATA_FLOAT,
            .real64F: CUDNN_DATA_DOUBLE,
            .real8I: CUDNN_DATA_INT8,
            .real32I: CUDNN_DATA_INT32,
            .real8U: CUDNN_DATA_UINT8
        ]
        assert(types[self] != nil, "Unknown cudnnDataType_t")
        return types[self]!
    }
}
