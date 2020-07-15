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
/// CublasHandle
/// creates and manages the lifetime of a cublas light handle
public final class CublasHandle 
{
    public let handle: cublasLtHandle_t

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
// MatmulAlgorithm
public final class MatmulAlgorithm {
    // properties
    public var desc: cublasLtMatmulAlgo_t

    // initializers
    @inlinable public init(
        cublas: CublasHandle,
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
        cublas: CublasHandle,
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