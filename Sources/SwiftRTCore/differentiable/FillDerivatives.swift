//******************************************************************************
// Copyright 2019 Google LLC
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

#if swift(>=5.3) && canImport(_Differentiation)

import Foundation
import _Differentiation

//==============================================================================
@derivative(of:concatenate)
@inlinable func _derivativeConcat<S, E>(
  _ tensors: [Tensor<S, E>],
  axis: Int = 0,
  into result: inout Tensor<S, E>
) -> (
  value: (),
  pullback: (inout Tensor<S, E>.TangentVector)
    -> Array<Tensor<S, E>>.TangentVector
) {
  let shapes = tensors.map { $0.shape }
  func pullback(_ resultTangent: inout Tensor<S, E>.TangentVector)
    -> Array<Tensor<S, E>>.TangentVector
  {
    // Fill `tensorTangents` with slices of `resultTangent` of shape
    // `tensorShapes[0]`, `tensorShapes[1]`, etc.
    var tensorTangents: [Tensor<S, E>] = []
    var lower = S.zero
    for shape in shapes {
      let upper = lower &+ shape
      tensorTangents.append(resultTangent[lower, upper])
      lower[axis] += upper[axis]
    }

    // Set `resultTangent` to zero.
    // Note: We can't use `fill(_:with:)` because `resultTangent` aliases
    // `tensorTangents`.
    // TODO: track and fix
    // Note: https://bugs.swift.org/browse/TF-1250 will allow us to make
    // this pullback more efficient. How:
    // - Set the wrt parameters and results to
    //     @differentiable(wrt: (tensors), results: (result))
    // - This makes `resultTangent` not be inout, so we don't need to set
    //   it any more.
    resultTangent = Tensor(
      zeros: resultTangent.shape,
      order: resultTangent.order)

    return Array.DifferentiableView(tensorTangents)
  }
  return (concatenate(tensors, axis: axis, into: &result), pullback)
}

//==============================================================================
// TODO: get this verified
@derivative(of:gather)
@usableFromInline func _derivativeGather<S, E>(
  from tensor: Tensor<S, E>,
  indices: TensorR1<DeviceIndex>,
  axis: Int = 0
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric {
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

#endif
