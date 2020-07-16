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
import Foundation
import CCuda

//==============================================================================
/// MatmulProperties
public struct MatmulProperties {

}

//==============================================================================
/// queryMatmulProperties
public func queryMatmulProperties<AE,BE,CE>(
    _ a: TensorR2<AE>, 
    transA: TransposeOp = .noTranspose,
    _ b: TensorR2<BE>, 
    transB: TransposeOp = .noTranspose,
    _ c: inout TensorR2<CE>,
    computeType: MatmulComputeType = .compute32F,
    scaleType: ScalarType = .real32F,
    maxAlgorithmsToTest: Int = Int.max,
    maxCombinationsToTest: Int = 100,
    timingRepeats: Int = 10
) -> MatmulProperties 
where AE: ScalarElement,
      BE: ScalarElement,
      CE: ScalarElement
{
    let splitKs = [2, 3, 4, 5, 6, 8, 12, 16, 32]
    let computeType = computeType
    let scaleType = ScalarType.real32F
    var operation = MatmulOperation(compute: computeType, scale: scaleType)
    operation.transA = transA
    operation.transB = transB
    print(operation)
    
    //
    var algorithmIds = [Int32](repeating: 0, count: maxAlgorithmsToTest)
    var algorithmsFound: Int32 = 0
    cudaCheck(cublasLtMatmulAlgoGetIds(
        Context.currentQueue.cublas.handle, 
        computeType.cublas,
        scaleType.cuda,
        AE.type.cuda,
        BE.type.cuda,
        CE.type.cuda,
        CE.type.cuda,
        Int32(maxAlgorithmsToTest),
        &algorithmIds, 
        &algorithmsFound))

    var combinationCount = 0

    for algo in 0..<Int(algorithmsFound) 
        where combinationCount <= maxCombinationsToTest 
    {

    }

    return MatmulProperties()
}

//==============================================================================
/// Structure to store information about different run trials
public struct MatmulPerformance {
    var algorithm: MatmulAlgorithm
    var status: cublasStatus_t
    var time: TimeInterval
    // actual memory workspace needed
    var workspaceSize: Int
    var mathMode: cublasMath_t
    var reductionScheme: cublasLtReductionScheme_t
    var customOption: Int32
    var wavesCount: Float
}

//==============================================================================
/// runCudaMatmul
/// runs and measures timing for the specified cublaslt matmul configuration 
public func runCudaMatmul(
    cublas: CublasHandle,
    operation: MatmulOperation,
    alpha: UnsafeRawPointer,
    A: UnsafeRawPointer,
    layoutA: MatrixLayout,
    B: UnsafeRawPointer,
    layoutB: MatrixLayout,
    beta: UnsafeRawPointer,
    C: UnsafeRawPointer,
    layoutC: MatrixLayout,
    D: UnsafeMutableRawPointer,
    layoutD: MatrixLayout,
    algorithm: MatmulAlgorithm,
    kernelRepeats: Int,
    workSpace: UnsafeMutableRawPointer,
    workSpaceSizeInBytes: Int,
    performanceResult: inout MatmulPerformance,
    stream: cudaStream_t,
    startEvent: inout cudaEvent_t,
    stopEvent: inout cudaEvent_t
) throws {
    // get algorithm heuristics
    let heur = try MatmulAlgorithmHeuristics(
        cublas: cublas,
        operation: operation,
        layoutA: layoutA,
        layoutB: layoutB,
        layoutC: layoutC,
        layoutD: layoutD,
        algorithm: algorithm)
    print(heur)
}

