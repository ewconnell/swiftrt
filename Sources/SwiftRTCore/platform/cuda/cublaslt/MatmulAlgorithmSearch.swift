//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LIDENSE-2.0
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

extension MatmulAlgorithm {
    //==========================================================================
    /// search
    ///
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    ///
    /// D = alpha*(A*B) + beta*(C),
    ///
    /// where A, B, and C are input matrices, and alpha and beta are
    /// input scalars.
    /// Note: matmul currently only supports the case where C == D 
    /// and Cdesc == Ddesc, so C is dropped from the api for now.
    ///
    /// Parameters:
    ///  - a: left hand side tensor
    ///  - transA: transpose operation to apply. Default: noTranspose
    ///  - b: left hand side tensor
    ///  - transB: transpose operation to apply. Default: noTranspose
    ///  - d: result
    ///  - accumulatorType: the accumulator precision to use
    ///  - scaleType: the scaling precision to use
    ///  - maxAlgorithmsToTest: the maximum number of algorithms to test
    ///  - maxTestVariations: the maximum number of variations per algorithm
    ///    to test.
    ///  - timingRepeats: the number of timing runs to perform per variation
    ///  - queue: the device queue to use
    ///
    public static func search<AE,BE,DE>(
        _ a: TensorR2<AE>, transA: TransposeOp = .noTranspose,
        _ b: TensorR2<BE>, transB: TransposeOp = .noTranspose,
        _ d: inout TensorR2<DE>,
        accumulatorType: MatmulAccumulatorType,
        scaleType: ScalarType,
        maxAlgorithmsToTest: Int = 100,
        maxTestVariations: Int = 100,
        timingRepeats: Int = 10,
        using queue: PlatformType.Device.Queue = Context.currentQueue
    ) -> MatmulProperties 
    where AE: ScalarElement, BE: ScalarElement, DE: ScalarElement
    {
        var combinationCount = 0
        let splitKs = [2, 3, 4, 5, 6, 8, 12, 16, 32]
        var operation = MatmulOperation(accumulatorType: accumulatorType, 
                                        scaleType: scaleType)
        operation.transA = transA
        operation.transB = transB

        // create layouts
        let layoutA = MatrixLayout(a)
        let layoutB = MatrixLayout(b)
        let layoutD = MatrixLayout(d)

        // get the available algorithm Ids for the data type combination    
        let algorithmIds = MatmulAlgorithm.getIds(
            maxIds: maxAlgorithmsToTest,
            accumulatorType: accumulatorType, 
            scaleType: scaleType,
            aType: AE.type, bType: BE.type, cType: DE.type, dType: DE.type)

        for algoId in algorithmIds  where combinationCount <= maxTestVariations 
        {
            let algo = MatmulAlgorithm(
                algoId: algoId, 
                accumulatorType: accumulatorType, 
                scaleType: scaleType,
                aType: AE.type, bType: BE.type, cType: DE.type, dType: DE.type)

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
                                    // configure the algorithm candidate
                                    let algorithm = MatmulAlgorithm(
                                        algoId: algoId,
                                        accumulatorType: accumulatorType,
                                        scaleType: scaleType,
                                        aType: AE.type,
                                        bType: BE.type,
                                        cType: DE.type,
                                        dType: DE.type,
                                        using: queue)

                                    // validate algo and get the workspace size
                                    let heur = MatmulAlgorithmHeuristics(
                                        cublas: queue.cublas,
                                        operation: operation,
                                        layoutA: layoutA,
                                        layoutB: layoutB,
                                        layoutC: layoutD,
                                        layoutD: layoutD,
                                        algorithm: algorithm)
                                    print(heur)

                                    // create the workspace

                                    // run the test
                                }
                            }
                        }
                    }
                }
            }
        }

        return MatmulProperties()
    }
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
