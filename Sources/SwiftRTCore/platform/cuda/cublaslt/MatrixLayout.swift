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

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init<E: ScalarElement>(_ tensor: TensorR2<E>) {
        var temp: cublasLtMatrixLayout_t?
        cudaCheck(cublasLtMatrixLayoutCreate(
            &temp, E.type.cuda, tensor.shape[0], tensor.shape[1],
            leadingDimension))
        desc = temp!
    }

    @inlinable deinit {
        cudaCheck(cublasLtMatrixLayoutDestroy(desc))
    }

    //--------------------------------------------------------------------------
    /// getAttribute
    @inlinable public func getAttribute<T>(
        _ attr: cublasLtMatrixLayoutAttribute_t, 
        _ value: inout T
    ) {
        var written = 0
        cudaCheck(cublasLtMatrixLayoutGetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value), &written))
    }

    /// setAttribute
    @inlinable public func setAttribute<T>(
        _ attr: cublasLtMatrixLayoutAttribute_t,
         _ value: inout T
    ) {
        cudaCheck(cublasLtMatrixLayoutSetAttribute(
            desc, attr, &value, MemoryLayout.size(ofValue: value)))
    }

    //--------------------------------------------------------------------------
    /// Specifies the data precision type
    @inlinable public var type: ScalarType {
        get {
            var value = CUDA_R_32F
            getAttribute(CUBLASLT_MATRIX_LAYOUT_TYPE, &value)
            return ScalarType(value)
        }
        set {
            var value = newValue.cuda
            setAttribute(CUBLASLT_MATRIX_LAYOUT_TYPE, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Specifies the data precision type
    @inlinable public var order: Order {
        get {
            var value = CUBLASLT_ORDER_ROW
            getAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, &value)
            return Order(value)
        }
        set {
            var value = newValue.cublas
            setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Describes the number of rows in the matrix. Only values
    /// that can be expressed as Int32 are supported
    @inlinable public var rows: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_ROWS, &value)
            return value
        }
        set {
            assert(newValue > 0 && newValue <= Int32.max)
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_ROWS, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Describes the number of cols in the matrix. Only values
    /// that can be expressed as Int32 are supported
    @inlinable public var cols: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_COLS, &value)
            return value
        }
        set {
            assert(newValue > 0 && newValue <= Int32.max)
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_COLS, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// The leading dimension of the matrix. For `Order.col` this is the
    /// stride (in elements) of matrix column. See also cublasLtOrder_t.
    ///  - Currently only non-negative values are supported.
    ///  - Must be large enough so that matrix memory locations are not
    ///    overlapping (e.g., greater or equal to `rows` in case of `Order.col`
    @inlinable public var leadingDimension: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_LD, &value)
            return value
        }
        set {
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_LD, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Number of matmul operations to perform in the batch. Default value is 1
    @inlinable public var batchCount: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &value)
            return value
        }
        set {
            assert(newValue > 0 && newValue <= Int32.max)
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Stride (in elements) to the next matrix for the strided batch
    /// operation. Default value is 0.
    @inlinable public var batchStride: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &value)
            return value
        }
        set {
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &value)
        }
    }

    //--------------------------------------------------------------------------
    /// Stride (in bytes) to the imaginary plane for planar complex layout. 
    /// Default value is 0, indicating that the layout is regular
    /// (real and imaginary parts of complex numbers are interleaved in
    /// memory for each element).
    @inlinable public var complexPlaneOffset: Int {
        get {
            var value = 0
            getAttribute(CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &value)
            return value
        }
        set {
            var value = newValue
            setAttribute(CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &value)
        }
    }
}

//==============================================================================
// Order
// mapping to cublas values
extension Order {
    @inlinable public init(_ type: cublasLtOrder_t) {
        switch type {
        case CUBLASLT_ORDER_COL: self = .col
        case CUBLASLT_ORDER_ROW: self = .row
        case CUBLASLT_ORDER_COL32: self = .colTiled32
        case CUBLASLT_ORDER_COL4_4R2_8C: self = .colTiledTC1
        case CUBLASLT_ORDER_COL32_2R_4R4: self = .colTiledTC2
        default: fatalError("unrecognized type")
        }
    }

    @inlinable public var cublas: cublasLtOrder_t {
        let types: [Order: cublasLtOrder_t] = [
            .col: CUBLASLT_ORDER_COL,
            .row: CUBLASLT_ORDER_ROW,
            .colTiled32: CUBLASLT_ORDER_COL32,
            .colTiledTC1: CUBLASLT_ORDER_COL4_4R2_8C,
            .colTiledTC2: CUBLASLT_ORDER_COL32_2R_4R4,
        ]        
        return types[self]!
    }
}
