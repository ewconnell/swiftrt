//******************************************************************************
// Copyright 2019 Google LLC
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
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
    //--------------------------------------------------------------------------
    @inlinable public func copy<S,E>(
        from a: Tensor<S,E>, 
        to b: inout Tensor<S,E>
    ) {
        if usesCpu || S.rank > 3 {
            cpu_copy(from: a, to: &b)
            return
        }

        // simple memcpy
        if a.order == b.order {
            if a.isContiguous && b.isContiguous {
                cudaCheck(cudaMemcpyAsync(
                    // dst
                    b.deviceReadWrite(using: self), 
                    // src
                    a.deviceRead(using: self), 
                    // count
                    MemoryLayout<E>.size * a.count, 
                    /// kind
                    cudaMemcpyDeviceToDevice, 
                    // stream
                    self.stream))
            } else {
                fatalError("strided copy not implemented yet")
            }
        } else {
            // swizzle transform
            // let transform = MatrixTransform(type: E.type)
            // let layoutA = MatrixLayout(a)
            // let layoutB = MatrixLayout(b)

            // cudaCheck(cublasLtMatrixTransform(
            //     // handle
            //     cublas.handle,
            //     // transformDesc
            //     transform.desc,
            //     // alpha
            //     E.onePointer,
            //     // A
            //     a.deviceRead(using: self),
            //     // layoutA
            //     layoutA.desc,
            //     // parameter B is not used for reording
            //     E.zeroPointer, nil, nil,
            //     // output
            //     b.deviceReadWrite(using: self),
            //     layoutB.desc,
            //     self.stream))
        }
    }
}