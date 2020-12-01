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

// JAX vmap
// https://jax.readthedocs.io/en/latest/jax.html#jax.vmap

//==============================================================================
public protocol VectorMappable {
  associatedtype M1: TensorShape
}

public protocol VectorMapped {
  associatedtype P1: TensorShape
}

extension Shape1: VectorMapped {
  public typealias P1 = Shape2
}

extension Shape2: VectorMappable, VectorMapped {
  public typealias M1 = Shape1
  public typealias P1 = Shape3
}

extension Shape3: VectorMappable, VectorMapped {
  public typealias M1 = Shape2
  public typealias P1 = Shape4
}

//==============================================================================
extension Tensor where Shape: VectorMappable {
  @inlinable public func asBatchedItem(
    along axis: Int
  ) -> Tensor<Shape.M1, TensorElement> {
    // get shape of batch item
    let batchCount = shape[axis]
    var itemShape = Shape.M1.zero
    var itemStrides = itemShape
    var j = 0
    for i in 0..<Shape.rank where i != axis {
      itemShape[j] = shape[i]
      itemStrides[j] = strides[i]
      j += 1
    }

    // create view for the batch item
    let itemCount = self.count / batchCount
    let itemSpanCount = itemShape.spanCount(stridedBy: itemStrides)

    return Tensor<Shape.M1, TensorElement>(
      shape: itemShape,
      strides: itemStrides,
      count: itemCount,
      storage: self.storage,
      storageBase: self.storageBase,
      spanCount: itemSpanCount,
      order: self.order,
      shared: self.isShared,
      batchCount: batchCount)
  }
}

//==============================================================================
/// vmap(tensor:axis:body
///
public func vmap<S, E>(
  _ t0: Tensor<S,E>,
  axis: Int = 0,
  outAxis: Int? = nil,
  _ body: (Tensor<S.M1, E>) -> Tensor<S.M1, E>
) -> Tensor<S, E> where S: VectorMappable, S.M1: VectorMapped {
  let outAxis = S.makePositive(axis: outAxis ?? axis)
  let item = t0.asBatchedItem(along: S.makePositive(axis: axis))
  let result = body(item)
  
  //-------------------------------------
  // expand
  // set the expanded axes
  var shape = S.zero
  var strides = S.zero

  // simple axis 0 case
  if outAxis == 0 {
    shape[0] = item.batchCount
    strides[0] = result.shape[0] * result.strides[0]
    for (i, j) in zip(1..<S.rank, 0..<S.M1.rank) {
      shape[i] = result.shape[j]
      strides[i] = result.strides[j]
    }
  } else {
//    shape[outAxis] = 1
//
//    var axis = Shape.rank - 1
//    var otherAxis = S.rank - 1
//    while axis >= 0 {
//      if shape[axis] == 1 {
//        if axis == Shape.rank - 1 {
//          // if the last dimension is expanded, then stride is 1
//          strides[axis] = 1
//        } else {
//          // if inserted, then compute stride
//          strides[axis] = shape[axis + 1] * strides[axis + 1]
//        }
//      } else {
//        // simply copy stride
//        shape[axis] = other.shape[otherAxis]
//        strides[axis] = other.strides[otherAxis]
//        otherAxis -= 1
//      }
//      axis -= 1
//    }
  }
  
  return Tensor<S, E>(
    shape: shape,
    strides: strides,
    count: shape.elementCount(),
    storage: result.storage,
    storageBase: result.storageBase,
    spanCount: shape.spanCount(stridedBy: strides),
    order: result.order,
    shared: result.isShared)
}

