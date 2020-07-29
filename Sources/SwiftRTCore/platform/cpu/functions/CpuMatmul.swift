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

//==============================================================================
/// CpuMatmul2
public final class CpuMatmul2<E>: DeviceMatmul2<E>
where E: StorageElement, E.Value: StorageElement & Numeric {}

//==============================================================================
/// DeviceMatmul2
public class DeviceMatmul2<E>: Logging
where E: StorageElement, E.Value: StorageElement & Numeric
{
    @inlinable public init() {}

    //--------------------------------------------------------------------------
    /// forward
    @inlinable public func forward(
        _ lhs: TensorR2<E>, _ transposeLhs: Bool,
        _ rhs: TensorR2<E>, _ transposeRhs: Bool,
        _ result: inout TensorR2<E>
    ) {
        let lhs = transposeLhs ? lhs.t : lhs
        let rhs = transposeRhs ? rhs.t : rhs
        assert(result.shape[0] == lhs.shape[0] &&
                result.shape[1] == rhs.shape[1],
               "matmul inner dimensions must be equal")
        //-------------------------------
        // simple place holder
        for r in 0..<result.shape[0] {
            let row = lhs[r, ...]
            for c in 0..<result.shape[1] {
                let col = rhs[..., c]
                result[r, c] = zip(row, col).reduce(into: 0) { $0 += $1.0 * $1.1 }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// backward
    @inlinable public func backward(
    ) {
        fatalError("abstract not implemented")
    }
}
