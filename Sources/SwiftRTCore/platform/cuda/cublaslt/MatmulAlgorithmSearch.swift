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
import SwiftRTCuda

//==============================================================================
/// MatmulProperties
public struct MatmulProperties {
    @inlinable public init() {}
}

extension MatmulAlgorithm {
    //==========================================================================
    /// query
    ///
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    ///
    /// D = alpha*(A*B) + beta*(C),
    ///
    /// where A, B, and C are input matrices, and alpha and beta are
    /// input scalars.
    /// Note: matmul currently only supports the case where C == D,
    /// so C is dropped from the api for now.
    ///
    /// Parameters:
    ///  - a: left hand tensor
    ///  - transA: transpose operation to apply. Default: noTranspose
    ///  - b: left hand tensor
    ///  - transB: transpose operation to apply. Default: noTranspose
    ///  - d: result
    ///  - accumulatorType: the accumulator precision to use
    ///  - scaleType: the scaling precision to use
    ///  - preferences: the algorithm query preferences
    ///  - maxResultCount: the maximum number of results to return
    ///  - queue: the device queue to use. Default is the current queue
    ///
    @inlinable public static func query<AE,BE,DE>(
        _ a: TensorR2<AE>, transA: TransposeOp = .noTranspose,
        _ b: TensorR2<BE>, transB: TransposeOp = .noTranspose,
        _ d: inout TensorR2<DE>,
        accumulatorType: MatmulAccumulatorType,
        scaleType: srtDataType,
        preferences: MatmulPreferences,
        maxResultCount: Int = 20,
        using queue: PlatformType.Device.Queue
    )  -> MatmulProperties {
        // TODO: figure out what scaleType depends on to expose properly
        let operation = MatmulOperation(accumulatorType: accumulatorType, 
                                        scaleType: scaleType)
        operation.transA = transA
        operation.transB = transB

        // create layouts
        let layoutA = MatrixLayout(a)
        let layoutB = MatrixLayout(b)
        let layoutD = MatrixLayout(d)
        print(layoutA)
        print(layoutB)
        print(layoutD)

        // do the query
        var returnAlgoCount: Int32 = 0
        var results = [MatmulAlgorithmHeuristicResult](
            repeating: MatmulAlgorithmHeuristicResult(), count: maxResultCount)
        let pResults = results.withUnsafeMutableBytes {
            $0.bindMemory(to: cublasLtMatmulHeuristicResult_t.self).baseAddress!
        }

        cudaCheck(cublasLtMatmulAlgoGetHeuristic(
            queue.cublas.handle,
            operation.desc,
            layoutA.desc,
            layoutB.desc,
            layoutD.desc,
            layoutD.desc,
            preferences.desc,
            Int32(maxResultCount),
            pResults,
            &returnAlgoCount))
        results = Array(results[..<Int(returnAlgoCount)])
        print(results)

        return MatmulProperties()
    }

    //==========================================================================
    /// search
    ///
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
    ///
    /// D = alpha*(A*B) + beta*(C),
    ///
    /// where A, B, and C are input matrices, and alpha and beta are
    /// input scalars.
    /// Note: matmul currently only supports the case where C == D, 
    /// so C is dropped from the api for now.
    ///
    /// Parameters:
    ///  - a: left hand tensor
    ///  - transA: transpose operation to apply. Default: noTranspose
    ///  - b: left hand tensor
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
    @inlinable public static func search<AE,BE,DE>(
        _ a: TensorR2<AE>, transA: TransposeOp = .noTranspose,
        _ b: TensorR2<BE>, transB: TransposeOp = .noTranspose,
        _ d: inout TensorR2<DE>,
        accumulatorType: MatmulAccumulatorType,
        scaleType: srtDataType,
        maxAlgorithmsToTest: Int = 100,
        maxTestVariations: Int = 100,
        timingRepeats: Int = 10,
        using queue: PlatformType.Device.Queue = Context.currentQueue
    ) -> MatmulProperties {
        // var combinationCount = 0
        // let splitKs = [2, 3, 4, 5, 6, 8, 12, 16, 32]
        // let operation = MatmulOperation(accumulatorType: accumulatorType, 
        //                                 scaleType: scaleType)
        // operation.transA = transA
        // operation.transB = transB

        // // create layouts
        // let layoutA = MatrixLayout(a)
        // let layoutB = MatrixLayout(b)
        // let layoutD = MatrixLayout(d)

        // // get the available algorithm Ids for the data type combination    
        // let algorithmIds = MatmulAlgorithm.getIds(
        //     maxIds: maxAlgorithmsToTest,
        //     accumulatorType: accumulatorType, 
        //     scaleType: scaleType,
        //     aType: AE.type, bType: BE.type, cType: DE.type, dType: DE.type)

        // for algoId in algorithmIds  where combinationCount <= maxTestVariations 
        // {
        //     let algo = MatmulAlgorithm(
        //         algoId: algoId, 
        //         accumulatorType: accumulatorType, 
        //         scaleType: scaleType,
        //         aType: AE.type, bType: BE.type, cType: DE.type, dType: DE.type)

        //     print("-----------------------------")
        //     print(algo)
        //     print("")
        //     print(algo.capsDescription)
        //     print("")

        //     // test each tile configuration
        //     for tileId in algo.tileIds {
        //         // test each stages configuraiton
        //         for stagesId in algo.stagesIds {
        //             // test each custom option
        //             for customOptionId in 0..<algo.customOptionCount {
        //                 // test each cta swizzling option
        //                 for swizzle in MatmulThreadSwizzling.allCases {
        //                     if algo.supportsSplitK {
        //                         for redScheme in MatmulReductionScheme.allCases {
        //                             // configure the algorithm candidate
        //                             let algorithm = MatmulAlgorithm(
        //                                 algoId: algoId,
        //                                 accumulatorType: accumulatorType,
        //                                 scaleType: scaleType,
        //                                 aType: AE.type,
        //                                 bType: BE.type,
        //                                 cType: DE.type,
        //                                 dType: DE.type,
        //                                 using: queue)

        //                             // validate algo and get the workspace size
        //                             let heur = MatmulAlgorithmHeuristicResult(
        //                                 cublas: queue.cublas,
        //                                 operation: operation,
        //                                 layoutA: layoutA,
        //                                 layoutB: layoutB,
        //                                 layoutC: layoutD,
        //                                 layoutD: layoutD,
        //                                 algorithm: algorithm)
        //                             print(heur)

        //                             // create the workspace

        //                             // run the test
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

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
