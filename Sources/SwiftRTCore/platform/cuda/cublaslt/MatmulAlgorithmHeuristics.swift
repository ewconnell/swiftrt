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
import SwiftRTCuda


//==============================================================================
/// MatmulAlgorithmHeuristicResult
/// This can throw if the parameter combination is not supported
public struct MatmulAlgorithmHeuristicResult: CustomStringConvertible
{
    public let heuristicResult: cublasLtMatmulHeuristicResult_t

    /// init
    /// this initializer is used to clear storage to receive values
    /// from heuristic algorithm query
    @inlinable public init() {
        heuristicResult = cublasLtMatmulHeuristicResult_t()
    }

    /// init
    /// used to set result values during algorithm search
    @inlinable public init(
        operation: MatmulOperation,
        layoutA: MatrixLayout,
        layoutB: MatrixLayout,
        layoutC: MatrixLayout,
        layoutD: MatrixLayout,
        algorithm: MatmulAlgorithm,
        using queue: Platform.Device.Queue = currentQueue
    ) {
        var temp = cublasLtMatmulHeuristicResult_t()
        cudaCheck(cublasLtMatmulAlgoCheck(
            queue.cublas.handle,
            operation.desc,
            layoutA.desc,
            layoutB.desc,
            layoutC.desc,
            layoutD.desc,
            &algorithm.desc, 
            &temp))
        heuristicResult = temp
    }

    @inlinable public var description: String {
        """
        MatmulAlgorithmHeuristicResult
        algorithm    : \(algorithm)
        workspaceSize: \(workspaceSize)
        isValid      : \(isValid)
        waves        : \(waves)
        """
    }

    @inlinable public var algorithm: MatmulAlgorithm {
        MatmulAlgorithm(heuristicResult.algo)
    }

    @inlinable public var workspaceSize: Int {
        heuristicResult.workspaceSize
    }

    /// `true` if the result is valid, `false` if there are no available
    /// algorithms that match the input requirements
    @inlinable public var isValid: Bool {
        heuristicResult.state == CUBLAS_STATUS_SUCCESS
    }

    /// Waves count is a device utilization metric. A value of 1.0 suggests
    /// that when the kernel is launched it will fully occupy the GPU. The
    /// closer to 1.0 the better
    @inlinable public var waves: Float {
        heuristicResult.wavesCount
    }
}