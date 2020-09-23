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
import SwiftRT

//==============================================================================
/// pmap
///
@inlinable public func pmap<S0,E0,S1,E1>(
    _ t0: inout Tensor<S0,E0>, axis axis0: Int = 0,
    _ t1: inout Tensor<S1,E1>, axis axis1: Int = 0,
    devices: [Int]? = nil,
    partitions: Int? = nil,
    synchronous: Bool = false,
    _ body: @escaping (inout Tensor<S0,E0>, inout Tensor<S1,E1>) -> Void
) {
    // create shared mutable views
    let st0 = t0.shared(using: currentQueue)
    let st1 = t1.shared(using: currentQueue)

    // determine the number of partitions
    let partitions = partitions ??
        ProcessInfo.processInfo.activeProcessorCount / 2
    assert(t0.shape[axis0] / partitions != 0, "too many partions")

    func execute(_ i: Int, _ p0: inout Tensor<S0,E0>, _ p1: inout Tensor<S1,E1>) {
        // execute with partitions
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
    }

    // distribute the work
    let group = DispatchGroup()
    
    for i in 0..<partitions {
        var p0 = st0.partition(i, axis0, partitions)
        var p1 = st1.partition(i, axis1, partitions)

        if synchronous {
            execute(i, &p0, &p1)
        } else {
            DispatchQueue.global().async(group: group) { execute(i, &p0, &p1) }
        }
    }
    group.wait()
}

//==============================================================================
// helpers
extension Tensor {
    @inlinable public func partition(_ index: Int, _ axis: Int, _ count: Int) -> Self {
        let size = shape[axis] / count
        assert(size > 0)
        var lower = Shape.zero
        var upper = shape
        lower[axis] = index * size
        // clamp for tensor shapes that are not multiples of size
        upper[axis] = Swift.min(shape[axis], lower[axis] + size)
        return createView(lower, upper, true)
    }
}

