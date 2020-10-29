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
/// gather(from:indices:axis:
/// consolidates the specified slices
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func gather<S, E>(
  from tensor: Tensor<S, E>,
  indices: TensorR1<DeviceIndex>,
  axis: Int = 0
) -> Tensor<S, E>
where S: TensorShape {
  let axis = axis < 0 ? axis + S.rank : axis
  var shape = tensor.shape
  shape[axis] = indices.count
  var result = Tensor<S, E>(shape: shape, order: tensor.order)
  var rlower = S.zero
  var tlower = S.zero
  var rupper = tensor.shape
  var tupper = tensor.shape

  for (ri, ti) in indices.enumerated() {
    rlower[axis] = Int(ri)
    rupper[axis] = rlower[axis] + 1
    tlower[axis] = Int(ti)
    tupper[axis] = tlower[axis] + 1
    result[rlower, rupper] = tensor[tlower, tupper]
  }
  return result
}

// TODO: get this verified
@derivative(of:gather)
@usableFromInline func _vjpGather<S, E>(
  from tensor: Tensor<S, E>,
  indices: TensorR1<DeviceIndex>,
  axis: Int = 0
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where S: TensorShape, E.Value: DifferentiableNumeric {
  let axis = axis < 0 ? axis + S.rank : axis
  let value = gather(from: tensor, indices: indices, axis: axis)
  let shape = tensor.shape
  return (
    value,
    {
      var result = Tensor<S, E>(zeros: shape, order: tensor.order)
      var rlower = S.zero
      var tlower = S.zero
      var rupper = shape
      var tupper = shape
      for (ti, ri) in indices.enumerated() {
        rlower[axis] = Int(ri)
        rupper[axis] = rlower[axis] + 1
        tlower[axis] = Int(ti)
        tupper[axis] = tlower[axis] + 1
        result[rlower, rupper] = $0[tlower, tupper]
      }
      return result
    }
  )
}

extension Tensor {
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func gathering(
    indices: TensorR1<DeviceIndex>,
    axis: Int = 0
  ) -> Self {
    gather(from: self, indices: indices, axis: axis)
  }
}
