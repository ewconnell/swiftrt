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

extension Tensor where Element: AdditiveArithmetic {
  //----------------------------------------------------------------------------
  // add
  // tensor + tensor
  @inlinable public static func + (lhs: Self, rhs: Self) -> Self {
    /// MAKE THIS GO AWAY!! assert(lhs.shape == rhs.shape) should be true
    /// Hack to work around AD zero materialization design problem
    if lhs.isZero {
      return rhs
    } else if rhs.isZero {
      return lhs
    } else {
      assert(lhs.shape == rhs.shape)
      var result = Tensor(like: lhs)
      currentQueue.add(lhs, rhs, &result)
      return result
    }
  }

  // tensor + Element
  @inlinable public static func + (lhs: Self, rhs: Element) -> Self {
    var out = Tensor(like: lhs)
    currentQueue.add(lhs, rhs, &out)
    return out
  }

  // tensor += Element
  @inlinable public static func += (lhs: inout Self, rhs: Element) {
    lhs = lhs + rhs
  }

  // Element + tensor
  @inlinable public static func + (lhs: Element, rhs: Self) -> Self {
    rhs + lhs
  }

  // VectorProtocol
  @inlinable public func adding(_ x: Element) -> Self {
    self + x
  }
}

extension Tensor where Element: AdditiveArithmetic {
  //----------------------------------------------------------------------------
  // subtract
  @inlinable public static func - (lhs: Self, rhs: Self) -> Self {
    assert(lhs.shape == rhs.shape)
    var result = Tensor(like: lhs)
    currentQueue.subtract(lhs, rhs, &result)
    return result
  }

  // tensor - Element
  @inlinable public static func - (lhs: Self, rhs: Element) -> Self {
    var out = Tensor(like: lhs)
    currentQueue.subtract(lhs, rhs, &out)
    return out
  }

  @inlinable public static func -= (lhs: inout Self, rhs: Element) {
    lhs = lhs - rhs
  }

  // Element - tensor
  @inlinable public static func - (lhs: Element, rhs: Self) -> Self {
    var out = Tensor(like: rhs)
    currentQueue.subtract(lhs, rhs, &out)
    return out
  }

  // VectorProtocol
  @inlinable public func subtracting(_ x: Element) -> Self {
    self - x
  }
}

//==============================================================================
/// mul

// tensor * tensor + Element
//    @differentiable(where Element: DifferentiableNumeric)
@inlinable public func multiply<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: Tensor<S, E>,
  add bias: E.Value
) -> Tensor<S, E> where E.Value: Numeric {
  assert(lhs.shape == rhs.shape)
  var out = Tensor(like: lhs)
  currentQueue.multiply(lhs, rhs, add: bias, &out)
  return out
}

@inlinable public func multiply<S, E>(
  _ lhs: Tensor<S, E>,
  _ rhs: Tensor<S, E>,
  add bias: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Numeric {
  assert(lhs.shape == rhs.shape && lhs.shape == bias.shape)
  var out = Tensor(like: lhs)
  currentQueue.multiply(lhs, rhs, add: bias, &out)
  return out
}

// TODO: Remove this when we find a better way to deal with PointwiseMultiplicative.
#if !canImport(TensorFlow)
  infix operator .*
#endif

extension Tensor where Element: Numeric {
  //----------------------------------------------------------------------------
  // mul
  // tensor * tensor
  @inlinable public static func * (lhs: Self, rhs: Self) -> Self {
    assert(lhs.shape == rhs.shape)
    var out = Tensor(like: lhs)
    currentQueue.mul(lhs, rhs, &out)
    return out
  }

  @inlinable public static func *= (lhs: inout Self, rhs: Self) {
    lhs = lhs * rhs
  }

  // tensor * Element
  @inlinable public static func * (lhs: Self, rhs: Element) -> Self {
    var out = Tensor(like: lhs)
    currentQueue.mul(lhs, rhs, &out)
    return out
  }

  @inlinable public static func *= (lhs: inout Self, rhs: Element) {
    lhs = lhs * rhs
  }

  // Element * tensor
  @inlinable public static func * (lhs: Element, rhs: Self) -> Self {
    var out = Tensor(like: rhs)
    currentQueue.mul(rhs, lhs, &out)
    return out
  }

  @inlinable public func scaled(by scalar: Element) -> Self {
    self * scalar
  }

  // TODO: this syntax is incorrect and is only here to conform to
  // PointwiseMultiplicative and should be removed
  @inlinable public static func .* (lhs: Self, rhs: Self) -> Self {
    lhs * rhs
  }
}

extension Tensor where Element: AlgebraicField {
  //----------------------------------------------------------------------------
  // div
  // tensor / tensor
  @inlinable public static func / (lhs: Self, rhs: Self) -> Self {
    assert(lhs.shape == rhs.shape)
    var result = Tensor(like: lhs)
    currentQueue.div(lhs, rhs, &result)
    return result
  }

  @inlinable public static func /= (lhs: inout Self, rhs: Self) {
    lhs = lhs / rhs
  }

  // tensor / Element
  @inlinable public static func / (lhs: Self, rhs: Element) -> Self {
    var result = Tensor(like: lhs)
    currentQueue.div(lhs, rhs, &result)
    return result
  }

  @inlinable public static func /= (lhs: inout Self, rhs: Element) {
    lhs = lhs / rhs
  }

  // Element / tensor
  @inlinable public static func / (lhs: Element, rhs: Self) -> Self {
    var result = Tensor(like: rhs)
    currentQueue.div(lhs, rhs, &result)
    return result
  }

  // PointwiseMultiplicative
  @inlinable public var reciprocal: Self {
    1 / self
  }
}

//==============================================================================
/// abs(x)
/// computes the absolute value of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func abs<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Comparable & SignedNumeric {
  var out = Tensor(like: x)
  currentQueue.abs(x, &out)
  return out
}

@inlinable public func abs<S, E>(
  _ x: Tensor<S, Complex<E>>
) -> Tensor<S, E> where E == E.Value, E.Value: Comparable & SignedNumeric {
  var out = Tensor<S, E>(shape: x.shape, order: x.order)
  currentQueue.abs(x, &out)
  return out
}

// Tensor extension to disambiguate with Swift.abs
extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
  // make glboal function visible for extension implementations
  @inlinable public func abs(_ x: Self) -> Self { SwiftRTCore.abs(x) }

  @inlinable public func abs() -> Self { abs(self) }
}

//==============================================================================
/// abs2(x)
/// computes the absolute value of Complex `x` squared
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func abs2<S, E>(
  _ x: Tensor<S, Complex<E>>
) -> Tensor<S, E> where E == E.Value, E.Value: Comparable & SignedNumeric {
  var out = Tensor<S, E>(shape: x.shape, order: x.order)
  currentQueue.abs2(x, &out)
  return out
}

//==============================================================================
/// acos(x)
/// computes the inverse cosine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func acos<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.acos(x, &out)
  return out
}

//==============================================================================
/// acosh(x)
/// computes the inverse hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func acosh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.acosh(x, &out)
  return out
}

//==============================================================================
/// asin(x)
/// computes the inverse sine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func asin<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.asin(x, &out)
  return out
}

//==============================================================================
/// asinh(x)
/// computes the inverse hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func asinh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.asinh(x, &out)
  return out
}

//==============================================================================
/// atan(x)
/// computes the inverse tangent of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func atan<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.atan(x, &out)
  return out
}

//==============================================================================
/// atanh(x)
/// computes the inverse hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func atanh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.atanh(x, &out)
  return out
}

//==============================================================================
/// atan2(y:x:
/// computes the arc tangent of a pair of values
/// - Parameter y: value tensor
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func atan2<S, E>(
  y: Tensor<S, E>,
  x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.atan2(y, x, &out)
  return out
}

//==============================================================================
/// cast(_:elementsTo:
/// casts elements of `tensor` to the output type
/// - Parameter tensor: value tensor
/// - Returns: out
@inlinable public func cast<S, E, OE>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE> where E.Value: BinaryInteger, OE.Value: BinaryFloatingPoint {
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

@inlinable public func cast<S, E, OE>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE> where E.Value: BinaryFloatingPoint, OE.Value: BinaryInteger {
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

@inlinable public func cast<S, E, OE>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE> where E.Value: Numeric, OE.Value == Bool {
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

@inlinable public func cast<S, E, OE>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE> where E.Value == Bool, OE.Value: Numeric {
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

@inlinable public func cast<S, E, OE, OR>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE>
where OE.Value == Complex<OR>, OR: BinaryFloatingPoint, E.Value: BinaryFloatingPoint {
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

@inlinable public func cast<S, E, ER, OE, OR>(
  _ tensor: Tensor<S, E>,
  elementsTo type: OE.Type
) -> Tensor<S, OE>
where
  E.Value == Complex<ER>, ER: BinaryFloatingPoint,
  OE.Value == Complex<OR>, OR: BinaryFloatingPoint
{
  var out = Tensor<S, OE>(shape: tensor.shape, order: tensor.order)
  currentQueue.cast(from: tensor, to: &out)
  return out
}

//==============================================================================
/// cos(x)
/// computes the cosine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func cos<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.cos(x, &out)
  return out
}

//==============================================================================
/// cosh(x)
/// computes the hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func cosh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.cosh(x, &out)
  return out
}

//==============================================================================
/// erf(x)
/// computes the error function of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func erf<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.erf(x, &out)
  return out
}

//==============================================================================
/// erfc(x)
/// computes the complementary error function of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func erfc<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.erfc(x, &out)
  return out
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func exp<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.exp(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @inlinable public func exp(_ x: Self) -> Self { SwiftRTCore.exp(x) }

  @inlinable public func exp() -> Self { exp(self) }
}

//==============================================================================
/// exp2(x)
/// Returns two raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func exp2<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.exp2(x, &out)
  return out
}

//==============================================================================
/// exp10(x)
/// Returns 10 raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: out
/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable public func exp10<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.exp10(x, &out)
  return out
}

//==============================================================================
/// expMinusOne(x)
/// computes the exponential minus one value of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func expMinusOne<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.expMinusOne(x, &out)
  return out
}

//==============================================================================
/// gamma(x)
/// computes the gamma of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func gamma<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.gamma(x, &out)
  return out
}

//==============================================================================
/// hypot(x:y:
/// calculate the length of the hypotenuse of a right triangle
/// - Parameter x: value tensor
/// - Parameter y: value tensor
/// - Returns: out
@inlinable public func hypot<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.hypot(x, y, &out)
  return out
}

//==============================================================================
/// log(x)
/// computes the log of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func log<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.log(x, &out)
  return out
}

@inlinable public func log2<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.log2(x, &out)
  return out
}

@inlinable public func log10<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.log10(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @inlinable public func log(_ x: Self) -> Self { SwiftRTCore.log(x) }

  @inlinable public func log() -> Self { log(self) }
}

//==============================================================================
/// log(onePlus x:
/// computes one plus the log of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func log<S, E>(
  onePlus x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.log(onePlus: x, &out)
  return out
}

//==============================================================================
/// logGamma(x)
/// computes the log gamma of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func logGamma<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.logGamma(x, &out)
  return out
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func neg<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: SignedNumeric {
  var out = Tensor(like: x)
  currentQueue.neg(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: SignedNumeric {
  // make glboal function visible for extension implementations
  @inlinable public static prefix func - (x: Self) -> Self { SwiftRTCore.neg(x) }

  @inlinable public func neg() -> Self { -self }
}

//==============================================================================
/// sin(x)
/// computes the sign of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func sin<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.sin(x, &out)
  return out
}

//==============================================================================
/// sinh(x)
/// computes the hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func sinh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.sinh(x, &out)
  return out
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func squared<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Numeric {
  var out = Tensor(like: x)
  currentQueue.squared(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Numeric {
  // make glboal function visible for extension implementations
  @inlinable public func squared(_ x: Self) -> Self { SwiftRTCore.squared(x) }

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
/// - Returns: out
@inlinable public func pow<S, E>(
  _ x: Tensor<S, E>,
  _ y: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  assert(x.shape == y.shape, _messageTensorShapeMismatch)
  var out = Tensor(like: x)
  currentQueue.pow(x, y, &out)
  return out
}

// pow(n
@inlinable public func pow<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.pow(x, n, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @inlinable public func pow(_ x: Self, _ y: Self) -> Self { SwiftRTCore.pow(x, y) }
}

//==============================================================================
/// root(x:n:
/// computes the nth root of `x`
/// - Parameter x: value tensor
/// - Parameter n: power
/// - Returns: out
@inlinable public func root<S, E>(
  _ x: Tensor<S, E>,
  _ n: Int
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.root(x, n, &out)
  return out
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func sqrt<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.sqrt(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @inlinable public func sqrt(_ x: Self) -> Self { SwiftRTCore.sqrt(x) }

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
  var out = Tensor(like: x)
  currentQueue.sign(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
  // make glboal function visible for extension implementations
  @inlinable public func sign(_ x: Self) -> Self { SwiftRTCore.sign(x) }

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
  var out = Tensor(like: x)
  currentQueue.sigmoid(x, &out)
  return out
}

// Tensor extension
extension Tensor where TensorElement.Value: Real {
  // make glboal function visible for extension implementations
  @inlinable public func sigmoid(_ x: Self) -> Self { SwiftRTCore.sigmoid(x) }

  @inlinable public func sigmoid() -> Self { sign(self) }
}

//==============================================================================
/// tan(x)
/// computes the tangent of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func tan<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.tan(x, &out)
  return out
}

//==============================================================================
/// tanh(x)
/// computes the hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: out
@inlinable public func tanh<S, E>(
  _ x: Tensor<S, E>
) -> Tensor<S, E> where E.Value: Real {
  var out = Tensor(like: x)
  currentQueue.tanh(x, &out)
  return out
}
