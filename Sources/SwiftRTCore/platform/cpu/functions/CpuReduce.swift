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

//==============================================================================
//
public typealias ReduceArg<E> = (index: Int, value: E.Value) where E: StorageElement

extension CpuQueue {
  //============================================================================
  @inlinable public func cpu_reduce<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ out: inout Tensor<S, E>,
    _ initialValue: E.Value,
    _ op: @escaping (E.Value, E.Value) -> E.Value
  ) {
    let axis = S.makePositive(axis: axis)
    assert(a.order == out.order)
    assert(axis >= 0 && axis < S.rank, "axis is out of range: \(axis)")
    assert(a.isContiguous, "input must be contiguous")

    if S.rank == 1 {
      out[out.startIndex] = a.buffer.reduce(into: initialValue) { $0 = op($0, $1) }
    } else {
      // the batch count is the product of the leading dimensions
      let batchCount = a.shape.reduce(range: 0..<axis, into: 1, &*=)
      let axisCount = a.shape[axis]
      
      // flatten the trailing dimensions and create batch views
      var elementCount = 1
      for i in (axis + 1)..<S.rank { elementCount &*= a.shape[i] }

      let batchA = TensorR2<E>(reshaping: a, to: Shape2(batchCount * axisCount, elementCount))
      let batchOut = TensorR2<E>(reshaping: out.shared(), to: Shape2(batchCount, elementCount))
      print(batchA)

      for bi in 0..<batchCount {
        let baseA = bi * axisCount
        // get the output vector and initialize with the first axis vector
        var outElements = batchOut[bi]
        mapOp(batchA[baseA], &outElements) { $0 }
        // reduce additional axis vectors using the specified op
        for ai in (baseA + 1)..<(baseA + axisCount) {
          let aElements = batchA[ai]
          print(aElements)
          print("before\n \(outElements)")
          mapOpNew(aElements, &outElements, op)
          print("after\n \(outElements)")
        }
      }
    }
  }
  
  //============================================================================
  @inlinable public func cpu_reduce<S, E>(
    _ a: Tensor<S, E>,
    _ axis: Int,
    _ arg: inout Tensor<S, Int32>,
    _ out: inout Tensor<S, E>,
    _ initialValue: E.Value,
    _ op: @escaping (ReduceArg<E>, ReduceArg<E>) -> ReduceArg<E>
  ) {
    let axis = S.makePositive(axis: axis)
    assert(axis >= 0 && axis < S.rank, "axis is out of range: \(axis)")
    assert(a.isContiguous, "input must be contiguous")

    if S.rank == 1 {
      let (index, value) = a.buffer.enumerated().reduce(into: (0, initialValue)) {
        $0 = op($0, $1)
      }
      arg[arg.startIndex] = Int32(index)
      out[out.startIndex] = value
    } else {
      // the batch count is the product of the leading dimensions
      var batchCount = 1
      for i in 0..<axis { batchCount &*= a.shape[i] }
    }
  }
}
