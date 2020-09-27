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

public enum PmapBound {
    case compute, bandwidth
}

//==============================================================================
/// pmap
///
@inlinable public func pmap<S0,E0,S1,E1>(
    _ t0: Tensor<S0,E0>, axis axis0: Int = 0,
    _ t1: inout Tensor<S1,E1>, axis axis1: Int = 0,
    _ partitions: Int? = nil,
    devices: [Int]? = nil,
    limitedBy: PmapBound = .bandwidth,
    _ body: @escaping (inout Tensor<S0,E0>, inout Tensor<S1,E1>) -> Void
) {
    // create shared mutable views
    let st1 = t1.shared(using: currentQueue)

    // determine the number of partitions
    // If not specified, then for bandwidth bound problems use the number
    // of physical cores, otherwise for compute bound problems use
    // the number of threads
    let partitions = min(partitions ?? {
        limitedBy == .compute ?
            ProcessInfo.processInfo.activeProcessorCount :
            ProcessInfo.processInfo.activeProcessorCount / 2
    }(), t0.shape[axis0])

    // execute partition
    func execute(_ p0: inout Tensor<S0,E0>, _ p1: inout Tensor<S1,E1>) {
        var tp0 = p0
        var tp1 = p1
        body(&tp0, &tp1)
        Thread.sleep(forTimeInterval: 0.1)
        p0.assign(tp0)
        p1.assign(tp1)
    }

    if partitions == 0 {
        var t0 = t0
        execute(&t0, &t1)
    } else {
        // distribute the work
        let group = DispatchGroup()
        for i in 0..<partitions {
            var p0 = t0.partition(i, axis0, partitions)
            var p1 = st1.partition(i, axis1, partitions)
            DispatchQueue.global().async(group: group) { execute(&p0, &p1) }
        }
        group.wait()
    }
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
    
    @inlinable public mutating func assign(_ other: Self) {
        assert(shape == other.shape)
        if !(storage === other.storage && storageBase == other.storageBase) {
            copy(from: other, to: &self)
        }
    }
}
