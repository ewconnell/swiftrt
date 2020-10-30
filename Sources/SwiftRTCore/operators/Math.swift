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

import Numerics
import _Differentiation

//==============================================================================
/// abs(x)
/// computes the absolute value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func abs<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Comparable & SignedNumeric {
  var result = Tensor(like: x)
  currentQueue.abs(x, &result)
  return result
}

@inlinable public func abs<S, E>(
  _ x: Tensor<S, Complex<E>>
) -> Tensor<S, E> where E == E.Value, E.Value: Comparable & SignedNumeric {
  var result = Tensor<S, E>(shape: x.shape, order: x.order)
  currentQueue.abs(x, &result)
  return result
}

@derivative(of:abs)
@usableFromInline func _vjpAbs<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & SignedNumeric {
  let signX = sign(x)
  return (abs(x), { $0 * signX })
}

// Tensor extension to disambiguate with Swift.abs
extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func abs(_ x: Self) -> Self { SwiftRTCore.abs(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func abs() -> Self { abs(self) }
}

//==============================================================================
/// acos(x)
/// computes the inverse cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acos<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.acos(x, &result)
  return result
}

@derivative(of:acos)
@usableFromInline func _vjpAcos<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (acos(x), { -$0 / sqrt(1 - x.squared()) })
}

//==============================================================================
/// acosh(x)
/// computes the inverse hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acosh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.acosh(x, &result)
  return result
}

@derivative(of:acosh)
@usableFromInline func _vjpAcosh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (acosh(x), { $0 / asinh(x) })
}

//==============================================================================
/// asin(x)
/// computes the inverse sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asin<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.asin(x, &result)
  return result
}

@derivative(of:asin)
@usableFromInline func _vjpAsin<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (asin(x), { $0 / sqrt(1 - x.squared()) })
}

//==============================================================================
/// asinh(x)
/// computes the inverse hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asinh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: DifferentiableNumeric & Real {
  var result = Tensor(like: x)
  currentQueue.asinh(x, &result)
  return result
}

@derivative(of:asinh)
@usableFromInline func _vjpAsinh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (asinh(x), { $0 / acosh(x) })
}

//==============================================================================
/// atan(x)
/// computes the inverse tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atan<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.atan(x, &result)
  return result
}

@derivative(of:atan)
@usableFromInline func _vjpAtan<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (atan(x), { $0 / (1 + x.squared()) })
}

//==============================================================================
/// atanh(x)
/// computes the inverse hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atanh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.atanh(x, &result)
  return result
}

@derivative(of:atanh)
@usableFromInline func _vjpAtanh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (atanh(x), { $0 / (1 - x.squared()) })
}

//==============================================================================
/// atan2(y:x:
/// computes the arc tangent of a pair of values
/// - Parameter y: value tensor
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atan2<S, E>(
  y: Tensor<S, E>,
  x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.atan2(y, x, &result)
  return result
}

@derivative(of:atan2)
@usableFromInline func _vjpAtan2<S, E>(
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

//==============================================================================
/// cast(from:to:
/// casts elements of `x` to the output type
/// - Parameter other: value tensor
/// - Returns: result
@inlinable public func cast<S, E, OE>(
  _ other: Tensor<S, OE>
) -> Tensor<S, E>
where
  E.Value: BinaryFloatingPoint,
  OE.Value: BinaryInteger
{
  var result = Tensor<S, E>(shape: other.shape)
  currentQueue.cast(from: other, to: &result)
  return result
}

@inlinable public func cast<S, E, OE>(
  _ other: Tensor<S, OE>
) -> Tensor<S, E>
where
  E.Value: BinaryInteger,
  OE.Value: BinaryFloatingPoint
{
  var result = Tensor<S, E>(shape: other.shape)
  currentQueue.cast(from: other, to: &result)
  return result
}

//==============================================================================
/// cos(x)
/// computes the cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cos<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.cos(x, &result)
  return result
}

@derivative(of:cos)
@usableFromInline func _vjpCos<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (cos(x), { -$0 * sin(x) })
}

//==============================================================================
/// cosh(x)
/// computes the hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cosh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.cosh(x, &result)
  return result
}

@derivative(of:cosh)
@usableFromInline func _vjpCosh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (cosh(x), { $0 * sinh(x) })
}

//==============================================================================
/// erf(x)
/// computes the error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erf<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.erf(x, &result)
  return result
}

@derivative(of:erf)
@usableFromInline func _vjpErf<S, E>(
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

//==============================================================================
/// erfc(x)
/// computes the complementary error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erfc<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.erfc(x, &result)
  return result
}

@derivative(of:erfc)
@usableFromInline func _vjpErfc<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.exp(x, &result)
  return result
}

@derivative(of:exp)
@usableFromInline func _vjpExp<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = exp(x)
  return (value, { $0 * value })
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func exp(_ x: Self) -> Self { SwiftRTCore.exp(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func exp() -> Self { exp(self) }
}

//==============================================================================
/// exp2(x)
/// Returns two raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp2<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.exp2(x, &result)
  return result
}

//==============================================================================
/// exp10(x)
/// Returns 10 raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: result
/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable public func exp10<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.exp10(x, &result)
  return result
}

//==============================================================================
/// expMinusOne(x)
/// computes the exponential minus one value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func expMinusOne<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.expMinusOne(x, &result)
  return result
}

@derivative(of:expMinusOne)
@usableFromInline func _vjpExpMinusOne<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let y = expMinusOne(x)
  return (y, { $0 * y })
}

//==============================================================================
/// gamma(x)
/// computes the gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func gamma<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.gamma(x, &result)
  return result
}

@derivative(of:gamma)
@usableFromInline func _vjpGamma<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// hypot(x:y:
/// calculate the length of the hypotenuse of a right triangle
/// - Parameter x: value tensor
/// - Parameter y: value tensor
/// - Returns: result
@inlinable public func hypot<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.hypot(x, y, &result)
  return result
}

@derivative(of:hypot)
@usableFromInline func _vjpHypot<S, E>(
  x: Tensor<S, E>,
  y: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// log(x)
/// computes the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func log<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.log(x, &result)
  return result
}

@derivative(of:log(_:))
@usableFromInline func _vjpLog<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (log(x), { $0 / x })
}

@inlinable public func log2<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.log2(x, &result)
  return result
}

@inlinable public func log10<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.log10(x, &result)
  return result
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func log(_ x: Self) -> Self { SwiftRTCore.log(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func log() -> Self { log(self) }
}

//==============================================================================
/// log(onePlus x:
/// computes one plus the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func log<S, E>(
  onePlus x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.log(onePlus: x, &result)
  return result
}

@derivative(of:log(onePlus:))
@usableFromInline func _vjpLogOnePlus<S, E>(
  onePlus x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// logGamma(x)
/// computes the log gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func logGamma<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.logGamma(x, &result)
  return result
}

@derivative(of:logGamma)
@usableFromInline func _vjpLogGamma<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func neg<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: SignedNumeric {
  var result = Tensor(like: x)
  currentQueue.neg(x, &result)
  return result
}

@derivative(of:neg)
@usableFromInline func _vjpNeg<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & SignedNumeric {
  (-x, { -$0 })
}

// Tensor extension
extension Tensor where TensorElement.Value: SignedNumeric {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public static prefix func - (x: Self) -> Self { SwiftRTCore.neg(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func neg() -> Self { -self }
}

//==============================================================================
/// sin(x)
/// computes the sign of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sin<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.sin(x, &result)
  return result
}

@derivative(of:sin)
@usableFromInline func _vjpSin<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (sin(x), { $0 * cos(x) })
}

//==============================================================================
/// sinh(x)
/// computes the hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sinh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.sinh(x, &result)
  return result
}

@derivative(of:sinh)
@usableFromInline func _vjpSinh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  (sinh(x), { $0 * cosh(x) })
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func squared<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Numeric {
  var result = Tensor(like: x)
  currentQueue.squared(x, &result)
  return result
}

@derivative(of:squared)
@usableFromInline func _vjpSquared<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>))
where E.Value: DifferentiableNumeric {
  (squared(x), { $0 * (x + x) })
}

// Tensor extension
extension Tensor where TensorElement.Value: Numeric {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func squared(_ x: Self) -> Self { SwiftRTCore.squared(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func squared() -> Self { squared(self) }
}

/// Numeric extension for scalar types
extension Numeric {
  @inlinable public func squared() -> Self { self * self }
}

//==============================================================================
/// pow(x)
/// computes elementwise `x` to the power of `y`
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable public func pow<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  assert(x.shape == y.shape, _messageTensorShapeMismatch)
  var result = Tensor(like: x)
  currentQueue.pow(x, y, &result)
  return result
}

@derivative(of:pow)
@usableFromInline func _vjpPow<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>, Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan  The S4TF version is too complex and needs to be rethought in
  // terms of SwiftRT syntax
  fatalError()
}

// pow(n
@inlinable public func pow<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.pow(x, n, &result)
  return result
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func pow(_ x: Self, _ y: Self) -> Self { SwiftRTCore.pow(x, y) }
}

//==============================================================================
/// root(x:n:
/// computes the nth root of `x`
/// - Parameter x: value tensor
/// - Parameter n: power
/// - Returns: result
@inlinable public func root<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.root(x, n, &result)
  return result
}

@derivative(of:root)
@usableFromInline func _vjpRoot<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> (Tensor<S, E>))
where E.Value: DifferentiableNumeric & Real {
  // Dan
  fatalError("Not implemented")
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sqrt<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.sqrt(x, &result)
  return result
}

@derivative(of:sqrt)
@usableFromInline func _vjpSqrt<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = sqrt(x)
  return (value, { $0 / (2 * value) })
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sqrt(_ x: Self) -> Self { SwiftRTCore.sqrt(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sqrt() -> Self { sqrt(self) }
}

//==============================================================================
/// sign(x)
///
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable public func sign<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Comparable & SignedNumeric {
  var result = Tensor(like: x)
  currentQueue.sign(x, &result)
  return result
}

@derivative(of:sign)
@usableFromInline func _vjpSign<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Comparable & SignedNumeric {
  // TODO: measure performance between repeating( and zeros(
  (sign(x), { _ in repeating(0, like: x) })
}

// Tensor extension
extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sign(_ x: Self) -> Self { SwiftRTCore.sign(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sign() -> Self { sign(self) }
}

//==============================================================================
/// sigmoid(x)
/// Returns the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable public func sigmoid<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.sigmoid(x, &result)
  return result
}

@derivative(of:sigmoid)
@usableFromInline func _vjpSigmoid<S, E>(
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

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sigmoid(_ x: Self) -> Self { SwiftRTCore.sigmoid(x) }

  @differentiable(where TensorElement.Value: DifferentiableNumeric)
  @inlinable public func sigmoid() -> Self { sign(self) }
}

//==============================================================================
/// tan(x)
/// computes the tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func tan<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.tan(x, &result)
  return result
}

@derivative(of:tan)
@usableFromInline func _vjpTan<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = tan(x)
  return (value, { $0 * (1 + value.squared()) })
}

//==============================================================================
/// tanh(x)
/// computes the hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func tanh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var result = Tensor(like: x)
  currentQueue.tanh(x, &result)
  return result
}

@derivative(of:tanh)
@usableFromInline func _vjpTanh<S, E>(
  _ x: Tensor<S, E>
) -> (value: Tensor<S, E>, pullback: (Tensor<S, E>) -> Tensor<S, E>)
where E.Value: DifferentiableNumeric & Real {
  let value = tanh(x)
  return (value, { $0 * (1 - value.squared()) })
}
