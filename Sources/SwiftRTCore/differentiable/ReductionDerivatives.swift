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

import Numerics
import _Differentiation

@derivative(of:sum(_:axis:))
@usableFromInline func _derivativeSum<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric {
  let xshape = x.shape
  return (sum(x, axis: axis), { Tensor<S, E>(repeating: $0, to: xshape) })
}

@derivative(of:mean)
@usableFromInline func _derivativeMean<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & AlgebraicField {
  let count = E.Value(exactly: x.count)!
  return (
    x.mean(axis: axis),
    { [xshape = x.shape] in
      Tensor<S, E>(repeating: $0, to: xshape) / count
    }
  )
}

@derivative(of:prod)
@usableFromInline func _derivativeProd<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric {
  (
    prod(x, axis: axis),
    { [xshape = x.shape] in
      Tensor<S, E>(repeating: $0, to: xshape)
    }
  )
}

@derivative(of:prodNonZeros)
@usableFromInline func _derivativeProdNonZeros<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric {
  // REVIEW: this is probably wrong
  // Dan
  let value = prodNonZeros(x, axis: axis)
  return (
    value,
    { [xshape = x.shape] in
      Tensor<S, E>(repeating: $0, to: xshape)
    }
  )
}

@derivative(of:min)
@usableFromInline func _derivativeMin<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & ComparableLimits {
  // Dan
  fatalError()
}

@derivative(of:max)
@usableFromInline func _derivativeMax<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & ComparableLimits {
  // Dan
  fatalError()
}

@derivative(of:abssum)
@usableFromInline func _derivativeAbsSum<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & SignedNumeric & Comparable {
  // Dan
  fatalError()
}

#endif
