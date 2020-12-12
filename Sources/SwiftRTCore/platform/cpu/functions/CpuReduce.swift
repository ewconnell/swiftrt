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

extension CpuFunctions where Self: DeviceQueue {
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
    assert(a.order == .row && a.order == out.order, "only row order is implemented")
    assert(axis >= 0 && axis < S.rank, "axis is out of range: \(axis)")

    //-----------------------------------
    // get buffers and iterators
    let aBuffer = a.read(using: currentQueue)
    let aStorageBase = a.storageBase
    func getIterA(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: aBuffer, storageBase: aStorageBase,
        startIndex: at, count: count)
    }

    let oBuffer = out.readWrite(using: currentQueue)
    let oStorageBase = out.storageBase
    func getIterO(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: oBuffer, storageBase: oStorageBase,
        startIndex: at, count: count)
    }
    
    //-----------------------------------
    // batch shape
    let bShape = Shape3(
      // flatten the leading axes to form a batch
      a.shape.reduce(range: 0..<axis, into: 1, &*=),
      // the number of items per reduction set
      a.shape[axis],
      // the number of elements per reduction item
      a.shape.reduce(range: (axis + 1)..<S.rank, into: 1, &*=))
    let bStrides = bShape.strides(for: a.order)
    var batchBaseA = 0, batchBaseO = 0
    
    func execute() {
      //-----------------------------------
      for _ in 0..<bShape[0] {
        // start the reduction at the batch base
        var axisItem = batchBaseA
        
        // create the output batch item iterator
        var iterO = getIterO(at: batchBaseO, count: bShape[2])
        
        // if the axis items form a column vector then
        // perform a single reduction on all axis items
        if bShape[2] == 1 {
          let iterA = getIterA(at: axisItem, count: bShape[1])
          iterO[iterO.startIndex] = iterA.reduce(into: initialValue, op)
          
        } else {
          // initialize output by copying first axis item elements
          let iterA = getIterA(at: axisItem, count: bShape[2])
          zip(iterO.indices, iterA).forEach { iterO[$0] = $1 }
          
          // reduce additional axis items using the specified op
          for _ in 1..<bShape[1] {
            axisItem += bStrides[1]
            let iterA = getIterA(at: axisItem, count: bShape[2])
            zip(iterO.indices, iterA).forEach { op(&iterO[$0], $1) }
          }
        }
        
        // move to next batch item
        batchBaseA += bStrides[0]
        batchBaseO += bStrides[1]
      }
    }
    
    if mode == .sync {
      execute()
    } else {
      queue.async(group: group) {
        execute()
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
    _ op: @escaping (inout ReduceArg<E>, ReduceArg<E>) -> Void
  ) {
    let axis = S.makePositive(axis: axis)
    assert(a.isContiguous, "input storage must be contiguous")
    assert(a.order == .row && a.order == out.order && a.order == arg.order,
           "only row order is implemented")
    assert(axis >= 0 && axis < S.rank, "axis is out of range: \(axis)")
    assert({
      var expected = a.shape
      expected[axis] = 1
      return out.shape == expected
    }(), "invalid output shape")
    assert(arg.shape == out.shape, "arg and out must be the same shape")

    //-----------------------------------
    // get buffers and iterators
    let aBuffer = a.read(using: currentQueue)
    let aStorageBase = a.storageBase
    func getIterA(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: aBuffer, storageBase: aStorageBase,
        startIndex: at, count: count)
    }

    let argBuffer = arg.readWrite(using: currentQueue)
    let argStorageBase = arg.storageBase
    func getIterArg(at: Int, count: Int) -> BufferElements<Int32> {
      BufferElements<Int32>(
        buffer: argBuffer, storageBase: argStorageBase,
        startIndex: at, count: count)
    }

    let outBuffer = out.readWrite(using: currentQueue)
    let outStorageBase = out.storageBase
    func getIterOut(at: Int, count: Int) -> BufferElements<E> {
      BufferElements<E>(
        buffer: outBuffer, storageBase: outStorageBase,
        startIndex: at, count: count)
    }
    
    //-----------------------------------
    // batch shape
    let bShape = Shape3(
      // flatten the leading axes to form a batch
      a.shape.reduce(range: 0..<axis, into: 1, &*=),
      // the number of items per reduction set
      a.shape[axis],
      // the number of elements per reduction item
      a.shape.reduce(range: (axis + 1)..<S.rank, into: 1, &*=))
    let bStrides = bShape.strides(for: a.order)
    var batchBaseA = 0, batchBaseO = 0
    
    func execute() {
      //-----------------------------------
      for _ in 0..<bShape[0] {
        // start the reduction at the batch base
        var axisItem = batchBaseA
        
        // create the output batch item iterator
        var iterArg = getIterArg(at: batchBaseO, count: bShape[2])
        var iterOut = getIterOut(at: batchBaseO, count: bShape[2])

        // if the axis items form a column vector then
        // perform a single reduction on all axis items
        if bShape[2] == 1 {
          let iterA = getIterA(at: axisItem, count: bShape[1])
          let result = iterA.enumerated().reduce(into: (0, initialValue), op)
          iterArg[iterArg.startIndex] = Int32(result.index)
          iterOut[iterOut.startIndex] = result.value
          
        } else {
          // initialize output by copying first axis item elements
          let iterA = getIterA(at: axisItem, count: bShape[2])
          iterArg.indices.forEach { iterArg[$0] = 0 }
          zip(iterOut.indices, iterA).forEach { iterOut[$0] = $1 }

          // reduce additional axis items using the specified op
          for i in 1..<bShape[1] {
            axisItem += bStrides[1]
            let iterA = getIterA(at: axisItem, count: bShape[2])
            zip(iterOut.indices, iterA).forEach {
              var result: ReduceArg<E> = (Int(iterArg[$0]), iterOut[$0])
              op(&result, (i, $1))
              iterArg[$0] = Int32(result.index)
              iterOut[$0] = result.value
            }
          }
        }
        
        // move to next batch item
        batchBaseA += bStrides[0]
        batchBaseO += bStrides[1]
      }
    }
    
    if mode == .sync {
      execute()
    } else {
      queue.async(group: group) {
        execute()
      }
    }
  }
}
