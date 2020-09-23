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

extension Tensor {
    @inlinable public func partition(index: Int, axis: Int, size: Int) -> Self {
        var lower = Shape.zero
        var upper = shape
        lower[axis] = index * size
        // clamp for tensor shapes that are not multiples of size
        upper[axis] = Swift.min(shape[axis], lower[axis] + size)
        return createView(lower, upper, true)
    }
}

//==============================================================================
/// pmap
///
@inlinable public func pmap<S0,E0,S1,E1>(
    _ t0: inout Tensor<S0,E0>, axis axis0: Int = 0,
    _ t1: inout Tensor<S1,E1>, axis axis1: Int = 0,
    devices: [Int]? = nil,
    partitionCount: Int? = nil,
    _ body: @escaping (inout Tensor<S0,E0>, inout Tensor<S1,E1>) -> Void
) {
    let partitionCount = partitionCount ?? ProcessInfo.processInfo.activeProcessorCount
    assert(t0.shape[axis0] / partitionCount != 0, "too many partions")
    let st0 = t0.shared(using: currentQueue)
    let st1 = t1.shared(using: currentQueue)
    let p0Size = st0.shape[axis0] / partitionCount
    let p1Size = st1.shape[axis1] / partitionCount
    let group = DispatchGroup()
    
    for i in 0..<partitionCount {
//        DispatchQueue.global().async(group: group) {
            // execute with partitions
            var p0 = st0.partition(index: i, axis: axis0, size: p0Size)
            var p1 = st1.partition(index: i, axis: axis1, size: p1Size)
            var tp0 = p0
            var tp1 = p1
            body(&tp0, &tp1)
            
            // copy back if overwritten by user function
            // e.g. `p0 = p0 + 1`
            if tp0.storage !== p0.storage {
                copy(from: tp0, to: &p0)
            }
            if p1.storage !== st1.storage {
                copy(from: tp1, to: &p1)
            }
//        }
    }
    group.wait()
}
