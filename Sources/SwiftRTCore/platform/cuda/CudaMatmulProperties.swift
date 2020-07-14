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
public func queryMatmulProperties<E>(
    _ a: TensorR2<E>, 
    _ transA: Bool,
    _ b: TensorR2<E>,
    _ transB: Bool,
    _ c: inout TensorR2<E>
) -> MatmulProperties {


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
    cublas: CublasLtHandle,
    operation: MatmulDescriptor,
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

