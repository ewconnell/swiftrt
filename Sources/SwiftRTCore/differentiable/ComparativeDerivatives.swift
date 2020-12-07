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

import _Differentiation

//==============================================================================
// min
@derivative(of:min)
@usableFromInline func _derivativeMin<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: Tensor<S, E>
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>)
) where E.Value: DifferentiableNumeric & Comparable {
  (
    value: min(lhs, rhs),
    {
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMin(lhs, rhs, $0, &resultTrue, &resultFalse)
      return (resultTrue, resultFalse)
    }
  )
}

// tensor Element
@derivative(of:min)
@usableFromInline func _derivativeMin<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: E.Value
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, E.Value)
) where E.Value: DifferentiableNumeric & Comparable {
  let value = min(lhs, rhs)
  return (
    value,
    { v in
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMin(lhs, rhs, v, &resultTrue, &resultFalse)
      return (resultTrue, resultFalse.sum().element)
    }
  )
}

@derivative(of:min, wrt: lhs)
@usableFromInline func _derivativeMin<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: E.Value
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>
) where E.Value: DifferentiableNumeric & Comparable {
  let value = max(lhs, rhs)
  return (
    value,
    { v in
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMin(lhs, rhs, v, &resultTrue, &resultFalse)
      return resultTrue
    }
  )
}

//==============================================================================
// max
// tensor tensor
@derivative(of:max)
@usableFromInline func _derivativeMax<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: Tensor<S, E>
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>)
) where E.Value: DifferentiableNumeric & Comparable {
  (
    value: max(lhs, rhs),
    {
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMax(lhs, rhs, $0, &resultTrue, &resultFalse)
      return (resultTrue, resultFalse)
    }
  )
}

// tensor Element
@derivative(of:max)
@usableFromInline func _derivativeMax<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: E.Value
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, E.Value)
) where E.Value: DifferentiableNumeric & Comparable {
  let value = max(lhs, rhs)
  return (
    value,
    { v in
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMax(lhs, rhs, v, &resultTrue, &resultFalse)
      return (resultTrue, resultFalse.sum().element)
    }
  )
}

@derivative(of:max,wrt: lhs)
@usableFromInline func _derivativeMax<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: E.Value
) -> (
  value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>
) where E.Value: DifferentiableNumeric & Comparable {
  let value = max(lhs, rhs)
  return (
    value,
    { v in
      var resultTrue = Tensor(like: lhs)
      var resultFalse = Tensor(like: lhs)
      currentQueue.vjpMax(lhs, rhs, v, &resultTrue, &resultFalse)
      return resultTrue
    }
  )
}

#endif
