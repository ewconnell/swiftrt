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
// DeviceQueue functions with default cpu delegation
extension CudaQueue 
{
    public func matmul2<E>(type: E.Type) -> DeviceMatmul2<E>
    where E: StorageElement, E.Value: ScalarElement & Numeric {
        CudaMatmul2<E>(queue: self)
    }
}

//==============================================================================
/// CudaMatmul2
public final class CudaMatmul2<E>: DeviceMatmul2<E>
where E: StorageElement, E.Value: ScalarElement & Numeric
{
    // properties
    public let queue: CudaQueue
    // public let properties: MatmulProperties

    //--------------------------------------------------------------------------
    /// init
    @inlinable public init(queue: CudaQueue) {
        self.queue = queue
    }

    //--------------------------------------------------------------------------
    /// forward
    // assert(result.shape[0] == lhs.shape[0] &&
    //         result.shape[1] == rhs.shape[1],
    //        "matmul inner dimensions must be equal")
    @inlinable public override func forward(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) {
        do {
            try tune(lhs, transposeLhs, rhs, transposeRhs, &result)

        } catch {
            writeLog("\(error)")
            // TODO: is there a better way to handle this??
            fatalError("unrecoverable error")
        }
    }
    
    //--------------------------------------------------------------------------
    /// backward
    @inlinable public override func backward(
    ) {
        fatalError("abstract not implemented")
    }
}

//==============================================================================
public extension CudaMatmul2 
{
    @inlinable func tune(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) throws {
        // let operationDesc = MatmulDescriptor(accumulatorType: CUBLAS_COMPUTE_32F,
        //                                      scaleType: CUDA_R_32F)
    }
}