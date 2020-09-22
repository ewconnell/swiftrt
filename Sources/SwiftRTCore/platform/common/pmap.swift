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

public struct Partition<Shape: TensorShape> {
    public let axis: Int
    public let step: Int
    public let size: Int
    public var lower: Shape
    public var upper: Shape
    @inlinable public init(_ shape: Shape, _ axis: Int, _ count: Int) {
        self.axis = axis
        size = shape[axis]
        step = size / count
        lower = Shape.zero
        upper = shape
        upper[axis] = step
    }
    
    @inlinable public var next: Self {
        var n = self
        n.lower[axis] = n.upper[axis]
        n.upper[axis] = min(size, n.upper[axis] + step)
        return n
    }
}

//==============================================================================
/// pmap
///
@inlinable public func pmap<S0,E0,S1,E1>(
    _ t0: inout Tensor<S0,E0>, axis axis0: Int = 0,
    _ t1: inout Tensor<S1,E1>, axis axis1: Int = 0,
    devices: [Int]? = nil,
    partitions: Int? = nil,
    _ body: (inout Tensor<S0,E0>, inout Tensor<S1,E1>) -> Void
) {
    let count = partitions ?? currentDevice.queues.count
    var r0 = Partition(t0.shape, axis0, count)
    var r1 = Partition(t1.shape, axis1, count)

    for i in 0..<count {
        var p0 = t0.createView(r0.lower, r0.upper, true)
        var p1 = t1.createView(r1.lower, r1.upper, true)
        using(device: 0, queue: i) {
            body(&p0, &p1)
        }
        r0 = r0.next
        r1 = r1.next
    }
}
