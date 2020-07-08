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

// ===------------------------------------------------------------------------------------------===//
// Free-function-style differential operators
// ===------------------------------------------------------------------------------------------===//

import _Differentiation

// Value with gradient

@inlinable
public func valueWithGradient<T, Shape, Element>(
  at x: T,
  in f: @differentiable (T) -> Tensor<Shape, Element>
) -> (value: Tensor<Shape, Element>, gradient: T.TangentVector)
where T: Differentiable, Element: DifferentiableElement {
  let (y, pullback) = valueWithPullback(at: x, in: f)
  precondition(
    y.shape.scalarCount == 0,
    """
    The function being differentiated produced a tensor with shape \(y.shape). \
    You can only compute the gradient of functions that return scalar values.
    """)
  return (value: y, gradient: pullback(Tensor<Shape, Element>(1)))
}

@inlinable
public func valueWithGradient<T, U, Shape, Element>(
  at x: T,
  _ y: U,
  in f: @differentiable (T, U) -> Tensor<Shape, Element>
) -> (value: Tensor<Shape, Element>, gradient: (T.TangentVector, U.TangentVector))
where T: Differentiable, U: Differentiable, Element: DifferentiableElement {
  let (y, pullback) = valueWithPullback(at: x, y, in: f)
  precondition(
    y.shape.scalarCount == 0,
    """
    The function being differentiated produced a tensor with shape \(y.shape). \
    You can only compute the gradient of functions that return scalar values.
    """)
  return (value: y, gradient: pullback(Tensor<Shape, Element>(1)))
}

@inlinable
public func valueWithGradient<T, U, V, Shape, Element>(
  at x: T,
  _ y: U,
  _ z: V,
  in f: @differentiable (T, U, V) -> Tensor<Shape, Element>
) -> (value: Tensor<Shape, Element>, gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
where T: Differentiable, U: Differentiable, V: Differentiable, Element: DifferentiableElement {
  let (y, pullback) = valueWithPullback(at: x, y, z, in: f)
  precondition(
    y.shape.scalarCount == 0,
    """
    The function being differentiated produced a tensor with shape \(y.shape). \
    You can only compute the gradient of functions that return scalar values.
    """)
  return (value: y, gradient: pullback(Tensor<Shape, Element>(1)))
}

// Value with gradient (curried)

@inlinable
public func valueWithGradient<T, Shape, Element>(
  of f: @escaping @differentiable (T) -> Tensor<Shape, Element>
) -> (T) -> (value: Tensor<Shape, Element>, gradient: T.TangentVector)
where T: Differentiable, Element: DifferentiableElement {
  return { x in valueWithGradient(at: x, in: f) }
}

@inlinable
public func valueWithGradient<T, U, Shape, Element>(
  of f: @escaping @differentiable (T, U) -> Tensor<Shape, Element>
) -> (T, U) -> (value: Tensor<Shape, Element>, gradient: (T.TangentVector, U.TangentVector))
where T: Differentiable, U: Differentiable, Element: DifferentiableElement {
  return { x, y in valueWithGradient(at: x, y, in: f) }
}

@inlinable
public func valueWithGradient<T, U, V, Shape, Element>(
  of f: @escaping @differentiable (T, U, V) -> Tensor<Shape, Element>
) -> (T, U, V) -> (
  value: Tensor<Shape, Element>,
  gradient: (T.TangentVector, U.TangentVector, V.TangentVector)
)
where T: Differentiable, U: Differentiable, V: Differentiable, Element: DifferentiableElement {
  return { x, y, z in valueWithGradient(at: x, y, z, in: f) }
}

// Gradient

@inlinable
public func gradient<T, Shape, Element>(
  at x: T,
  in f: @differentiable (T) -> Tensor<Shape, Element>
) -> T.TangentVector where T: Differentiable, Element: DifferentiableElement {
  return valueWithGradient(at: x, in: f).1
}

@inlinable
public func gradient<T, U, Shape, Element>(
  at x: T,
  _ y: U,
  in f: @differentiable (T, U) -> Tensor<Shape, Element>
) -> (T.TangentVector, U.TangentVector)
where T: Differentiable, U: Differentiable, Element: DifferentiableElement {
  return valueWithGradient(at: x, y, in: f).1
}

@inlinable
public func gradient<T, U, V, Shape, Element>(
  at x: T,
  _ y: U,
  _ z: V,
  in f: @differentiable (T, U, V) -> Tensor<Shape, Element>
) -> (T.TangentVector, U.TangentVector, V.TangentVector)
where T: Differentiable, U: Differentiable, V: Differentiable, Element: DifferentiableElement {
  return valueWithGradient(at: x, y, z, in: f).1
}

// Gradient (curried)

@inlinable
public func gradient<T, Shape, Element>(
  of f: @escaping @differentiable (T) -> Tensor<Shape, Element>
) -> (T) -> T.TangentVector where T: Differentiable, Element: DifferentiableElement {
  return { x in gradient(at: x, in: f) }
}

@inlinable
public func gradient<T, U, Shape, Element>(
  of f: @escaping @differentiable (T, U) -> Tensor<Shape, Element>
) -> (T, U) -> (T.TangentVector, U.TangentVector)
where T: Differentiable, U: Differentiable, Element: DifferentiableElement {
  return { x, y in gradient(at: x, y, in: f) }
}

@inlinable
public func gradient<T, U, V, Shape, Element>(
  of f: @escaping @differentiable (T, U, V) -> Tensor<Shape, Element>
) -> (T, U, V) -> (T.TangentVector, U.TangentVector, V.TangentVector)
where T: Differentiable, U: Differentiable, V: Differentiable, Element: DifferentiableElement {
  return { x, y, z in gradient(at: x, y, z, in: f) }
}
