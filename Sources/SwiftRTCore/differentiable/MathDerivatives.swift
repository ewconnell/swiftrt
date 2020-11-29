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

#if swift(>=5.3) && canImport(_Differentiation)

import _Differentiation

extension Tensor where Element: DifferentiableNumeric & AdditiveArithmetic {
  //----------------------------------------------------------------------------
  // add
  @derivative(of:+)
  @usableFromInline static func _derivativeAdd(_ lhs: Self, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Self, Self)
  ) {
    (lhs + rhs, { ($0, $0) })
  }
  
  @derivative(of:+)
  @usableFromInline static func _derivativeAdd(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> (Self, Element)
  ) {
    (lhs + rhs, { ($0, $0.sum().element) })
  }
  
  @derivative(of:+, wrt: lhs)
  @usableFromInline static func _derivativeAdd(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs + rhs, { $0 })
  }
  
  @derivative(of:+)
  @usableFromInline static func _derivativeAdd(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Element, Self)
  ) {
    (lhs + rhs, { ($0.sum().element, $0) })
  }
  
  @derivative(of:+,wrt: rhs)
  @usableFromInline static func _derivativeAdd(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs + rhs, { $0 })
  }
}

extension Tensor where Element: DifferentiableNumeric & SignedNumeric {
  //----------------------------------------------------------------------------
  // subtract
  @derivative(of:-)
  @usableFromInline static func _derivativeSubtract(_ lhs: Self, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Self, Self)
  ) {
    (lhs - rhs, { ($0, -$0) })
  }
  
  @derivative(of:-)
  @usableFromInline static func _derivativeSubtract(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> (Self, Element)
  ) {
    (lhs + rhs, { ($0, $0.sum().element) })
  }
  
  @derivative(of:-, wrt: lhs)
  @usableFromInline static func _derivativeSubtract(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs - rhs, { $0 })
  }
  
  @derivative(of:-)
  @usableFromInline static func _derivativeSubtract(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Element, Self)
  ) {
    (lhs + rhs, { ($0.sum().element, -$0) })
  }
  
  @derivative(of:-, wrt: rhs)
  @usableFromInline static func _derivativeSubtract(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs - rhs, { -$0 })
  }
}

extension Tensor where Element: DifferentiableNumeric & Numeric {
  //----------------------------------------------------------------------------
  // multiply
  @derivative(of:*)
  @usableFromInline static func _derivativeMultiply(_ lhs: Self, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Self, Self)
  ) where Element: DifferentiableNumeric {
    (lhs * rhs, { v in (v * rhs, v * lhs) })
  }

  @derivative(of:*)
  @usableFromInline static func _derivativeMultiply(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> (Self, Element)
  ) {
    (lhs * rhs, { ($0 * rhs, ($0 * lhs).sum().element) })
  }
  
  @derivative(of:*, wrt: lhs)
  @usableFromInline static func _derivativeMultiply(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs * rhs, { $0 * rhs })
  }

  @derivative(of:*)
  @usableFromInline static func _derivativeMultiply(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Element, Self)
  ) {
    (lhs * rhs, { (($0 * rhs).sum().element, $0 * lhs) })
  }
  
  @derivative(of:*,wrt: rhs)
  @usableFromInline static func _derivativeMultiply(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs * rhs, { lhs * $0 })
  }
}

extension Tensor where Element: DifferentiableNumeric & AlgebraicField {
  //--------------------------------------------------------------------------
  // div
  @derivative(of:/)
  @usableFromInline static func _derivativeDivide(_ lhs: Self, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Self, Self)
  ) {
    (lhs / rhs, { ($0 / rhs, -lhs / rhs.squared() * $0) })
  }
  
  @derivative(of:/)
  @usableFromInline static func _derivativeDivide(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> (Self, Element)
  ) {
    (lhs / rhs, { ($0 / rhs, ($0 * -lhs / rhs.squared()).sum().element) })
  }
  
  @derivative(of:/, wrt: lhs)
  @usableFromInline static func _derivativeDivide(_ lhs: Self, _ rhs: Element) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs / rhs, { $0 / rhs })
  }
  
  @derivative(of:/)
  @usableFromInline static func _derivativeDivide(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> (Element, Self)
  ) {
    (lhs / rhs, { (($0 / rhs).sum().element, $0 * -lhs / rhs.squared()) })
  }
  
  @derivative(of:/, wrt: rhs)
  @usableFromInline static func _derivativeDivide(_ lhs: Element, _ rhs: Self) -> (
    value: Self, pullback: (Self) -> Self
  ) {
    (lhs / rhs, { -lhs / rhs.squared() * $0 })
  }
}

//==============================================================================

@derivative(of:abs)
@usableFromInline func _derivativeAbs<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & SignedNumeric {
  let signX = sign(x)
  return (abs(x), { $0 * signX })
}

@derivative(of:acos)
@usableFromInline func _derivativeAcos<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (acos(x), { -$0 / sqrt(1 - x.squared()) })
}

@derivative(of:acosh)
@usableFromInline func _derivativeAcosh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (acosh(x), { $0 / asinh(x) })
}

@derivative(of:asin)
@usableFromInline func _derivativeAsin<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (asin(x), { $0 / sqrt(1 - x.squared()) })
}

@derivative(of:asinh)
@usableFromInline func _derivativeAsinh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (asinh(x), { $0 / acosh(x) })
}

@derivative(of:atan)
@usableFromInline func _derivativeAtan<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (atan(x), { $0 / (1 + x.squared()) })
}

@derivative(of:atanh)
@usableFromInline func _derivativeAtanh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (atanh(x), { $0 / (1 - x.squared()) })
}

@derivative(of:atan2)
@usableFromInline func _derivativeAtan2<S, E>(
  y: Tensor<S, E>,
  x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  let value = atan2(y: y, x: x)
  return (
    value,
    { v in
      let gradInv = v / ((x * x) + (y * y))
      return (x * gradInv, -y * gradInv)
    }
  )
}

@derivative(of:cos)
@usableFromInline func _derivativeCos<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (cos(x), { -$0 * sin(x) })
}

@derivative(of:cosh)
@usableFromInline func _derivativeCosh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (cosh(x), { $0 * sinh(x) })
}

@derivative(of:erf)
@usableFromInline func _derivativeErf<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = erf(x)
  return (
    value,
    { v in
      return v * (2 / E.Value.pi.squareRoot()) * exp(-(x * x))
    }
  )
}

@derivative(of:erfc)
@usableFromInline func _derivativeErfc<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:exp)
@usableFromInline func _derivativeExp<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = exp(x)
  return (value, { $0 * value })
}

@derivative(of:expMinusOne)
@usableFromInline func _derivativeExpMinusOne<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let y = expMinusOne(x)
  return (y, { $0 * y })
}

@derivative(of:gamma)
@usableFromInline func _derivativeGamma<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:hypot)
@usableFromInline func _derivativeHypot<S, E>(
  x: Tensor<S, E>,
  y: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:log(_:))
@usableFromInline func _derivativeLog<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (log(x), { $0 / x })
}

@derivative(of:log(onePlus:))
@usableFromInline func _derivativeLogOnePlus<S, E>(
  onePlus x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:logGamma)
@usableFromInline func _derivativeLogGamma<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:neg)
@usableFromInline func _derivativeNeg<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & SignedNumeric {
  (-x, { -$0 })
}

@derivative(of:sin)
@usableFromInline func _derivativeSin<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (sin(x), { $0 * cos(x) })
}

@derivative(of:sinh)
@usableFromInline func _derivativeSinh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (sinh(x), { $0 * cosh(x) })
}

@derivative(of:squared)
@usableFromInline func _derivativeSquared<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>))
where E.Value: DifferentiableNumeric {
  (squared(x), { $0 * (x + x) })
}

@derivative(of:pow)
@usableFromInline func _derivativePow<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan  The S4TF version is too complex and needs to be rethought in
  // terms of SwiftRT syntax
  fatalError()
}

@derivative(of:root)
@usableFromInline func _derivativeRoot<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

@derivative(of:sqrt)
@usableFromInline func _derivativeSqrt<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = sqrt(x)
  return (value, { $0 / (2 * value) })
}

@derivative(of:sign)
@usableFromInline func _derivativeSign<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & SignedNumeric {
  // TODO: measure performance between repeating( and zeros(
  (sign(x), { _ in repeating(0, like: x) })
}

@derivative(of:sigmoid)
@usableFromInline func _derivativeSigmoid<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (
    sigmoid(x),
    { v in
      // Dan
      fatalError()
    }
  )
}

@derivative(of:tan)
@usableFromInline func _derivativeTan<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = tan(x)
  return (value, { $0 * (1 + value.squared()) })
}

@derivative(of:tanh)
@usableFromInline func _derivativeTanh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = tanh(x)
  return (value, { $0 * (1 - value.squared()) })
}

//==============================================================================
// matmulGradients
// _vjpMatmul helper function
@usableFromInline func matmulGradients<E>(
  _ out: TensorR2<E>,
  _ lhs: TensorR2<E>, _ transposeLhs: Bool,
  _ rhs: TensorR2<E>, _ transposeRhs: Bool
) -> (TensorR2<E>, TensorR2<E>)
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  let (lhsGrad, rhsGrad): (TensorR2<E>, TensorR2<E>)
  switch (transposeLhs, transposeRhs) {
  case (false, false):
    lhsGrad = matmul(out, transposed: false, rhs, transposed: true)
    rhsGrad = matmul(lhs, transposed: true, out, transposed: false)
  case (false, true):
    lhsGrad = matmul(out, rhs)
    rhsGrad = matmul(lhs, transposed: true, out, transposed: false)
  case (true, false):
    lhsGrad = matmul(out, transposed: false, rhs, transposed: true)
    rhsGrad = matmul(lhs, out)
  case (true, true):
    lhsGrad = matmul(out, transposed: true, rhs, transposed: true)
    rhsGrad = matmul(lhs, transposed: true, out, transposed: true)
  }
  return (lhsGrad, rhsGrad)
}

@derivative(of:matmul)
@usableFromInline func _vjpMatmul<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR2<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  (
    matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
    { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) }
  )
}

@derivative(of:matmul,wrt: lhs)
@usableFromInline func _vjpMatmulWrtLhs<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  (
    matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
    { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs).0 }
  )
}

@derivative(of:matmul,wrt: rhs)
@usableFromInline func _vjpMatmulWrtRhs<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  (
    matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
    { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs).1 }
  )
}

@derivative(of:matmul)
@usableFromInline func _vjpMatmul<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
  bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  fatalError()
  //    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
  //     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

@derivative(of:matmul,wrt: (lhs, bias))
@usableFromInline func _vjpMatmulWrtLhsBias<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
  bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  fatalError()
  //    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
  //     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

@derivative(of:matmul,wrt: (rhs, bias))
@usableFromInline func _vjpMatmulWrtRhsBias<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
  bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: StorageElement & DifferentiableNumeric {
  fatalError()
  //    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
  //     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

//@derivative(of: matmul)
//@usableFromInline func _vjpMatmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
//    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, TensorR2<E>))
//where S: TensorShape, E.Value: DifferentiableNumeric
//{
//    fatalError()
//}
//
//
//@derivative(of: matmul)
//@usableFromInline func _vjpMatmul<S,E>(
//    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (TensorR2<E>, Tensor<S,E>))
//where S: TensorShape, E.Value: DifferentiableNumeric
//{
//    fatalError()
//}
//
//@derivative(of: matmul)
//@usableFromInline func _vjpMatmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
//where S: TensorShape, E.Value: DifferentiableNumeric
//{
//    fatalError()
//}

#endif
