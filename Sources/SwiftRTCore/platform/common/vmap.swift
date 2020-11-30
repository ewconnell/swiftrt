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

import Foundation

extension Tensor {
  //========================================================================
  @inlinable public func vmapItem<Axes: TensorShape, ItemShape: TensorShape>(
    along axes: Axes
  ) -> Tensor<ItemShape, TensorElement> {
    assert(Axes.rank + ItemShape.rank == Shape.rank, "rank mismatch")
    assert({
      for i in 1..<Axes.rank where axes[i] <= axes[i-1] { return false }
      return true
    }(), "axes must be specified in ascending order")

    // extract item shape
    var s = shape
    axes.indices.forEach { s[axes[$0]] = 1 }
    var itemShape = ItemShape(), itemStrides = ItemShape()
    var j = 0
    for i in shape.indices where s[i] != 1 {
      itemShape[j] = shape[i]
      itemStrides[j] = strides[i]
      j += 1
    }

    // create a view for the first batch item
    return Tensor<ItemShape, TensorElement>(
      shape: itemShape,
      strides: itemStrides,
      count: itemShape.elementCount(),
      storage: self.storage,
      storageBase: self.storageBase,
      spanCount: itemShape.spanCount(stridedBy: itemStrides),
      order: self.order,
      shared: self.isShared,
      batchCount: 1,
      batchStride: 1
    )
  }
  
  @inlinable public init<Axes: TensorShape, ItemShape: TensorShape>(
    from batched: Tensor<ItemShape, TensorElement>,
    along axes: Axes
  ) {
    fatalError()
  }
}

//==========================================================================
/// vmap(tensor:axis:body
///

//--------------------------------------------------------------------------
// Rank 2
public func vmap<E>(
  _ tensor: TensorR2<E>,
  axis: Int = 0,
  outAxis: Int = 0,
  _ body: (TensorR1<E>) -> TensorR1<E>
) -> TensorR2<E> {
  return TensorR2<E>(from: body(tensor.vmapItem(along: Shape1(axis))), along: Shape1(outAxis))
}

//--------------------------------------------------------------------------
// Rank 3
public func vmap<E>(
  _ tensor: TensorR3<E>,
  axis: Int = 0,
  outAxis: Int = 0,
  _ body: (TensorR2<E>) -> TensorR2<E>
) -> TensorR3<E> {
  return TensorR3<E>(from: body(tensor.vmapItem(along: Shape1(axis))), along: Shape1(outAxis))
}

public func vmap<E>(
  _ tensor: TensorR3<E>,
  axes: Shape2.Tuple,
  outAxes: Shape2.Tuple? = nil,
  _ body: (TensorR1<E>) -> TensorR1<E>
) -> TensorR3<E> {
  return TensorR3<E>(
    from: body(tensor.vmapItem(along: Shape2(axes))),
    along: Shape2(outAxes ?? axes))
}
