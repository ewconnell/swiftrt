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
    _ op: @escaping (inout E.Value, E.Value) -> Void
  ) {
    let axis = S.makePositive(axis: axis)
    assert(a.isContiguous, "input storage must be contiguous")
    assert((a.order == .row || a.order == .col) && a.order == out.order)
    assert(axis >= 0 && axis < S.rank, "axis is out of range: \(axis)")

    //-----------------------------------
    // the item is dense so the element count equals the item stride
    let elementCount = a.strides[axis]
    // the number of axis items (sets of elements) along the axis to reduce
    let axisItemCount = a.shape[axis]
    let axisItemStride = a.strides[axis]
    // flatten the leading axes to form a batch
    let batchItemCount = a.shape.reduce(range: 0..<axis, into: 1, &*=)
    let batchItemStrideA = axisItemCount * axisItemStride
    let batchItemStrideO = axisItemStride
    
    //-----------------------------------
    // get buffers and iterators
    let aBuffer = a.read(using: self)
    func getIterA(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: aBuffer, storageBase: a.storageBase,
        startIndex: at, count: count)
    }

    let oBuffer = out.readWrite(using: self)
    func getIterO(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: oBuffer, storageBase: out.storageBase,
        startIndex: at, count: count)
    }

    //-----------------------------------
    // do the reductions
    var batchBaseA = 0, batchBaseO = 0
    for _ in 0..<batchItemCount {
      // start the reduction at the batch base
      var axisItem = batchBaseA
      
      // create the output batch item iterator
      var iterO = getIterO(at: batchBaseO, count: elementCount)

      // if the axis items form a column vector then
      // perform a single reduction on all axis items
      if elementCount == 1 {
        let iterA = getIterA(at: axisItem, count: axisItemCount)
        iterO[iterO.startIndex] = iterA.reduce(into: initialValue, op)
        
      } else {
        // initialize output by copying first axis item elements
        let iterA = getIterA(at: axisItem, count: elementCount)
        zip(iterO.indices, iterA).forEach { iterO[$0] = $1 }
        
        // reduce additional axis items using the specified op
        for _ in 1..<axisItemCount {
          axisItem += axisItemStride
          let iterA = getIterA(at: axisItem, count: elementCount)
          zip(iterO.indices, iterA).forEach { op(&iterO[$0], $1) }
        }
      }

      // move to next batch item
      batchBaseA += batchItemStrideA
      batchBaseO += batchItemStrideO
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
