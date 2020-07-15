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
/// MatmulComputeType
public enum MatmulComputeType {
    /// Float16 default
    case compute16F
    /// Float16 precise 
    case compute16FPrecise
    /// Float default
    case compute32F
    /// Float precise
    case compute32FPrecise
    /// Float fast, allows down-converting inputs to half or TF32
    case compute32FFast16F
    /// Float fast, allows down-converting inputs to bfloat16 or TF32
    case compute32FFast16BF
    /// Float fast, allows down-converting inputs to TF32
    case compute32FFastTF32
    /// Double default
    case compute64F
    /// Double precise
    case compute64FPrecise
    /// Int32 default
    case compute32I
    /// Int32 precise
    case compute32IPrecise
}

extension MatmulComputeType {
    public var cublas: cublasComputeType_t {
        switch self {
        case .compute16F: return CUBLAS_COMPUTE_16F
        case .compute16FPrecise: return CUBLAS_COMPUTE_16F_PEDANTIC
        case .compute32F: return CUBLAS_COMPUTE_32F
        case .compute32FPrecise: return CUBLAS_COMPUTE_32F_PEDANTIC
        case .compute32FFast16F: return CUBLAS_COMPUTE_32F_FAST_16F
        case .compute32FFast16BF: return CUBLAS_COMPUTE_32F_FAST_16BF
        case .compute32FFastTF32: return CUBLAS_COMPUTE_32F_FAST_TF32
        case .compute64F: return CUBLAS_COMPUTE_64F
        case .compute64FPrecise: return CUBLAS_COMPUTE_64F_PEDANTIC
        case .compute32I: return CUBLAS_COMPUTE_32I
        case .compute32IPrecise: return CUBLAS_COMPUTE_32I_PEDANTIC
        }
    }
}

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

    @inlinable public func setAttribute<T>(
        _ attr: cublasLtMatmulDescAttributes_t,
         _ value: inout T
    ) {
        do {
            try cudaCheck(status: cublasLtMatmulDescSetAttribute(
                desc, attr, &value, MemoryLayout.size(ofValue: value)))
        } catch {
            Context.currentQueue.writeLog("\(error)")
            fatalError()
        }
    }

    @inlinable public func getAttribute<T>(
        _ attr: cublasLtMatmulDescAttributes_t, 
        _ value: inout T
    ) {
        do {
            var written = 0
            try cudaCheck(status: cublasLtMatmulDescGetAttribute(
                desc, attr, &value,MemoryLayout.size(ofValue: value), &written))
        } catch {
            Context.currentQueue.writeLog("\(error)")
            fatalError()
        }
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
        do {
            // create the descriptor
            var temp: cublasLtMatrixLayout_t?
            try cudaCheck(status: cublasLtMatrixLayoutCreate(
                &temp, type, rows, cols, leadingDimension))
            desc = temp!
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cublasLtMatrixLayoutDestroy(desc))
        } catch {
            Context.currentQueue.writeLog("\(releaseString) \(error)")
        }
    }
}

//==============================================================================
// MatmulAlgorithm
public final class MatmulAlgorithm {
    // properties
    public var desc: cublasLtMatmulAlgo_t

    // initializers
    @inlinable public init(
        cublas: CublasLtHandle,
        computeType: cublasComputeType_t,
        scaleType: cudaDataType_t,
        aType: cudaDataType_t,
        bType: cudaDataType_t,
        cType: cudaDataType_t,
        dType: cudaDataType_t,
        algoId: Int32
    ) {
        do {
            // initialize the algorithm
            desc = cublasLtMatmulAlgo_t()
            try cudaCheck(status: cublasLtMatmulAlgoInit(
                cublas.handle, computeType, scaleType, aType, bType,
                cType, dType, algoId, &desc))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }
}

//==============================================================================
/// MatmulAlgorithmHeuristics
/// This can throw if the parameter combination is not supported
public final class MatmulAlgorithmHeuristics {
    // properties
    public let heuristicResult: cublasLtMatmulHeuristicResult_t

    // initializers
    @inlinable public init(
        cublas: CublasLtHandle,
        operation: MatmulOperation,
        layoutA: MatrixLayout,
        layoutB: MatrixLayout,
        layoutC: MatrixLayout,
        layoutD: MatrixLayout,
        algorithm: MatmulAlgorithm
    ) throws {
        var temp = cublasLtMatmulHeuristicResult_t()
        try cudaCheck(status: cublasLtMatmulAlgoCheck(
            cublas.handle,
            operation.desc,
            layoutA.desc,
            layoutB.desc,
            layoutC.desc,
            layoutD.desc,
            &algorithm.desc, 
            &temp))
        heuristicResult = temp
    }
}