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
    accumulatorType: MatmulAccumulatorType,
    scaleType: ScalarType,
    maxAlgorithmsToTest: Int = 100,
    maxCombinationsToTest: Int = 100,
    timingRepeats: Int = 10,
    using queue: PlatformType.Device.Queue = Context.currentQueue
) -> MatmulProperties 
where AE: ScalarElement,
      BE: ScalarElement,
      CE: ScalarElement
{
    var combinationCount = 0
    let splitKs = [2, 3, 4, 5, 6, 8, 12, 16, 32]
    var operation = MatmulOperation(accumulatorType: accumulatorType, 
                                    scaleType: scaleType)
    operation.transA = transA
    operation.transB = transB

    // get the available algorithm Ids for the data type combination    
    let algorithmIds = MatmulAlgorithm.getIds(
        maxIds: maxAlgorithmsToTest,
        accumulatorType: accumulatorType, 
        scaleType: scaleType,
        aType: AE.type, bType: BE.type, cType: CE.type, dType: CE.type)

    for algoId in algorithmIds  where combinationCount <= maxCombinationsToTest 
    {
        let algo = MatmulAlgorithm(
            algoId: algoId, 
            accumulatorType: accumulatorType, 
            scaleType: scaleType,
            aType: AE.type, bType: BE.type, cType: CE.type, dType: CE.type)

        print("-----------------------------")
        print(algo)
        print("")
        print(algo.capsDescription)
        print("")

        // test each tile configuration
        for tileId in algo.tileIds {
            // test each stages configuraiton
            for stagesId in algo.stagesIds {
                // test each custom option
                for customOptionId in 0..<algo.customOptionCount {
                    // test each cta swizzling option
                    for swizzle in MatmulThreadSwizzling.allCases {
                        if algo.supportsSplitK {
                            for redScheme in MatmulReductionScheme.allCases {
                                // configure the algorithm

                                // create the workspace

                                // run the test
                                let perf = runCudaMatmul(
                                    operation: operation,
                                    alpha: AE.onePointer,
                                    A: a.deviceRead(using: queue),
                                    layoutA: MatrixLayout(a), 
                                    B: b.deviceRead(using: queue), 
                                    layoutB: MatrixLayout(b), 
                                    beta: CE.onePointer, 
                                    C: c.deviceReadWrite(using: queue),
                                    layoutC: MatrixLayout(c), 
                                    D: c.deviceReadWrite(using: queue), 
                                    layoutD: MatrixLayout(c), 
                                    algorithm: algo, 
                                    kernelRepeats: timingRepeats, 
                                    workSpace: UnsafeMutableRawPointer, 
                                    workSpaceSizeInBytes: Int,
                                    using: queue)
                            }
                        }
                    }
                }
            }
        }
    }

    return MatmulProperties()
}

//==============================================================================
/// Structure to store information about different run trials
public struct MatmulPerformance {
    let algorithm: MatmulAlgorithm
    let status: cublasStatus_t
    let time: TimeInterval
    // actual memory workspace needed
    let workspaceSize: Int
    let mathMode: cublasMath_t
    let reductionScheme: cublasLtReductionScheme_t
    let customOption: Int32
    let wavesCount: Float
}

//==============================================================================
/// runCudaMatmul
/// runs and measures timing for the specified cublaslt matmul configuration 
public func runCudaMatmul(
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
    using queue: PlatformType.Device.Queue = Context.currentQueue
) -> MatmulPerformance {
    // get algorithm heuristics
    let heur = MatmulAlgorithmHeuristics(
        cublas: queue.cublas,
        operation: operation,
        layoutA: layoutA,
        layoutB: layoutB,
        layoutC: layoutC,
        layoutD: layoutD,
        algorithm: algorithm)
    print(heur)
    fatalError()
}

