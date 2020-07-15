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
// MatrixLayout
public final class MatrixLayout {
    // properties
    public let desc: cublasLtMatrixLayout_t

    // initializers
    @inlinable public init(
        type: cudaDataType,
        rows: UInt64,
        cols: UInt64,
        leadingDimension: Int64
    ) {
        var temp: cublasLtMatrixLayout_t?
        cudaCheck(cublasLtMatrixLayoutCreate(
            &temp, type, rows, cols, leadingDimension))
        desc = temp!
    }

    @inlinable deinit {
        cudaCheck(cublasLtMatrixLayoutDestroy(desc))
    }
}

