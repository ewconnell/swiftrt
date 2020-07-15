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
        var temp: cublasLtHandle_t?
        cudaCheck(cublasLtCreate(&temp))
        handle = temp!
    }

    @inlinable deinit {
        cudaCheck(cublasLtDestroy(handle))
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
    @inlinable public init(_ type: cublasComputeType_t) {
        switch type {
        case CUBLAS_COMPUTE_16F: self = .compute16F
        case CUBLAS_COMPUTE_16F_PEDANTIC: self = .compute16FPrecise
        case CUBLAS_COMPUTE_32F: self = .compute32F
        case CUBLAS_COMPUTE_32F_PEDANTIC: self = .compute32FPrecise
        case CUBLAS_COMPUTE_32F_FAST_16F: self = .compute32FFast16F
        case CUBLAS_COMPUTE_32F_FAST_16BF: self = .compute32FFast16BF
        case CUBLAS_COMPUTE_32F_FAST_TF32: self = .compute32FFastTF32
        case CUBLAS_COMPUTE_64F: self = .compute64F
        case CUBLAS_COMPUTE_64F_PEDANTIC: self = .compute64FPrecise
        case CUBLAS_COMPUTE_32I: self = .compute32I
        case CUBLAS_COMPUTE_32I_PEDANTIC: self = .compute32IPrecise
        default: fatalError("unrecognized cublasComputeType_t")
        }
    }

    @inlinable public var cublas: cublasComputeType_t {
        let types: [MatmulComputeType: cublasComputeType_t] = [
            .compute16F: CUBLAS_COMPUTE_16F,
            .compute16FPrecise: CUBLAS_COMPUTE_16F_PEDANTIC,
            .compute32F: CUBLAS_COMPUTE_32F,
            .compute32FPrecise: CUBLAS_COMPUTE_32F_PEDANTIC,
            .compute32FFast16F: CUBLAS_COMPUTE_32F_FAST_16F,
            .compute32FFast16BF: CUBLAS_COMPUTE_32F_FAST_16BF,
            .compute32FFastTF32: CUBLAS_COMPUTE_32F_FAST_TF32,
            .compute64F: CUBLAS_COMPUTE_64F,
            .compute64FPrecise: CUBLAS_COMPUTE_64F_PEDANTIC,
            .compute32I: CUBLAS_COMPUTE_32I,
            .compute32IPrecise: CUBLAS_COMPUTE_32I_PEDANTIC,
        ]        
        return types[self]!
    }
}

//==============================================================================
// MatmulAlgorithm
public final class MatmulAlgorithm 
{
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
        desc = cublasLtMatmulAlgo_t()
        cudaCheck(cublasLtMatmulAlgoInit(
            cublas.handle, computeType, scaleType, aType, bType,
            cType, dType, algoId, &desc))
    }
}

//==============================================================================
/// MatmulAlgorithmHeuristics
/// This can throw if the parameter combination is not supported
public final class MatmulAlgorithmHeuristics 
{
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
        cudaCheck(cublasLtMatmulAlgoCheck(
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