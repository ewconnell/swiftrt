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

//==============================================================================
/// abs(x)
/// computes the absolute value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func abs<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
where S: TensorShape, E.Value: Comparable & SignedNumeric
{
    var result = Tensor(like: x)
    Context.currentQueue.abs(x, &result)
    return result
}

@derivative(of: abs)
@usableFromInline func _vjpAbs<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape,
          E.Value: DifferentiableNumeric & Comparable & SignedNumeric
{
    let signX = sign(x)
    return (abs(x), { $0 * signX })
}

// Tensor extension to disambiguate with Swift.abs
public extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func abs(_ x: Self) -> Self { SwiftRTCore.abs(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func abs() -> Self { abs(self) }
}

//==============================================================================
/// acos(x)
/// computes the inverse cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.acos(x, &result)
    return result
}

@derivative(of: acos)
@usableFromInline func _vjpAcos<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (acos(x), { v in -v / sqrt(1 - x.squared()) })
}

//==============================================================================
/// acosh(x)
/// computes the inverse hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.acosh(x, &result)
    return result
}

@derivative(of: acosh)
@usableFromInline func _vjpAcosh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (acosh(x), { v in v / asinh(x) })
}

//==============================================================================
/// asin(x)
/// computes the inverse sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.asin(x, &result)
    return result
}

@derivative(of: asin)
@usableFromInline func _vjpAsin<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (asin(x), { v in v / sqrt(1 - x.squared()) })
}

//==============================================================================
/// asinh(x)
/// computes the inverse hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    var result = Tensor(like: x)
    Context.currentQueue.asinh(x, &result)
    return result
}

@derivative(of: asinh)
@usableFromInline func _vjpAsinh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (asinh(x), { v in v / acosh(x) })
}

//==============================================================================
/// atan(x)
/// computes the inverse tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atan(x, &result)
    return result
}

@derivative(of: atan)
@usableFromInline func _vjpAtan<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (atan(x), { v in v / (1 + x.squared()) })
}

//==============================================================================
/// atanh(x)
/// computes the inverse hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atanh(x, &result)
    return result
}

@derivative(of: atanh)
@usableFromInline func _vjpAtanh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (atanh(x), { v in v / (1 - x.squared()) })
}

//==============================================================================
/// atan2(y:x:
/// computes the arc tangent of a pair of values
/// - Parameter y: value tensor
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atan2<S,E>(y: Tensor<S,E>, x: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atan2(y, x, &result)
    return result
}

@derivative(of: atan2)
@usableFromInline func _vjpAtan2<S,E>(y: Tensor<S,E>, x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    // TODO
    fatalError("Not implemented")
}

//==============================================================================
/// cast(from:to:
/// casts elements of `x` to the output type
/// - Parameter other: value tensor
/// - Returns: result
@inlinable public func cast<S,E,OE>(_ other: Tensor<S,OE>) -> Tensor<S,E>
where S: TensorShape, E.Value: BinaryFloatingPoint, OE.Value: BinaryInteger
{
    var result = Tensor<S,E>(shape: other.shape)
    Context.currentQueue.cast(from: other, to: &result)
    return result
}

@inlinable public func cast<S,E,OE>(_ other: Tensor<S,OE>) -> Tensor<S,E>
where S: TensorShape, E.Value: BinaryInteger, OE.Value: BinaryFloatingPoint
{
    var result = Tensor<S,E>(shape: other.shape)
    Context.currentQueue.cast(from: other, to: &result)
    return result
}

//==============================================================================
/// cos(x)
/// computes the cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.cos(x, &result)
    return result
}

@derivative(of: cos)
@usableFromInline func _vjpCos<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (cos(x), { v in -v * sin(x) })
}

//==============================================================================
/// cosh(x)
/// computes the hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.cosh(x, &result)
    return result
}

@derivative(of: cosh)
@usableFromInline func _vjpCosh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (cosh(x), { v in v * sinh(x) })
}

//==============================================================================
/// erf(x)
/// computes the error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erf<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.erf(x, &result)
    return result
}

@derivative(of: erf)
@usableFromInline func _vjpErf<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// erfc(x)
/// computes the complementary error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erfc<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.erfc(x, &result)
    return result
}

@derivative(of: erfc)
@usableFromInline func _vjpErfc<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.exp(x, &result)
    return result
}

@derivative(of: exp)
@usableFromInline func _vjpExp<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    let value = exp(x)
    return (value, { v in value * v } )
}

// Tensor extension
public extension Tensor where TensorElement.Value: Real {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func exp(_ x: Self) -> Self { SwiftRTCore.exp(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func exp() -> Self { exp(self) }
}

//==============================================================================
/// exp2(x)
/// Returns two raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp2<S,E>(_ x: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.exp2(x, &result)
    return result
}

//==============================================================================
/// exp10(x)
/// Returns 10 raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: result
/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable public func exp10<S,E>(_ x: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.exp10(x, &result)
    return result
}

//==============================================================================
/// expMinusOne(x)
/// computes the exponential minus one value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func expMinusOne<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.expMinusOne(x, &result)
    return result
}

@derivative(of: expMinusOne)
@usableFromInline func _vjpExpMinusOne<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    let y = expMinusOne(x)
    return (y, { v in v * y })
}

//==============================================================================
/// gamma(x)
/// computes the gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func gamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.gamma(x, &result)
    return result
}

@derivative(of: gamma)
@usableFromInline func _vjpGamma<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// hypot(x:y:
/// calculate the length of the hypotenuse of a right triangle
/// - Parameter x: value tensor
/// - Parameter y: value tensor
/// - Returns: result
@inlinable public func hypot<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.hypot(x, y, &result)
    return result
}

@derivative(of: hypot)
@usableFromInline func _vjpHypot<S,E>(x: Tensor<S,E>, y: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    // TODO:
    fatalError("Not implemented")
}

//==============================================================================
/// log(x)
/// computes the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func log<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log(x, &result)
    return result
}

@derivative(of: log(_:))
@usableFromInline func _vjpLog<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (log(x), { v in v / x })
}

@inlinable public func log2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log2(x, &result)
    return result
}

@inlinable public func log10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log10(x, &result)
    return result
}

// Tensor extension
public extension Tensor where TensorElement.Value: Real {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func log(_ x: Self) -> Self { SwiftRTCore.log(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func log() -> Self { log(self) }
}

//==============================================================================
/// log(onePlus x:
/// computes one plus the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func log<S,E>(onePlus x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log(onePlus: x, &result)
    return result
}

@derivative(of: log(onePlus:))
@usableFromInline func _vjpLogOnePlus<S,E>(onePlus x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// logGamma(x)
/// computes the log gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func logGamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.logGamma(x, &result)
    return result
}

@derivative(of: logGamma)
@usableFromInline func _vjpLogGamma<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func neg<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: SignedNumeric
{
    var result = Tensor(like: x)
    Context.currentQueue.neg(x, &result)
    return result
}

@derivative(of: neg)
@usableFromInline func _vjpNeg<S,E>(_ x: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & SignedNumeric
{
    (-x, { v in -v })
}

// Tensor extension
public extension Tensor where TensorElement.Value: SignedNumeric {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable static prefix func - (x: Self) -> Self { SwiftRTCore.neg(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func neg() -> Self { -self }
}

//==============================================================================
/// sin(x)
/// computes the sign of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sin(x, &result)
    return result
}

@derivative(of: sin)
@usableFromInline func _vjpSin<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (sin(x), { v in v * cos(x) })
}

//==============================================================================
/// sinh(x)
/// computes the hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sinh(x, &result)
    return result
}

@derivative(of: sinh)
@usableFromInline func _vjpSinh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (sinh(x), { v in v * cosh(x) })
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func squared<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
where S: TensorShape, E.Value: Numeric
{
    var result = Tensor(like: x)
    Context.currentQueue.squared(x, &result)
    return result
}

@derivative(of: squared)
@usableFromInline func _vjpSquared<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric
{
    (squared(x), { v in v * (x + x) })
}

// Tensor extension
public extension Tensor where TensorElement.Value: Numeric {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func squared(_ x: Self) -> Self { SwiftRTCore.squared(x) }
    
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func squared() -> Self { squared(self) }
}

/// Numeric extension for scalar types
public extension Numeric {
    @inlinable func squared() -> Self { self * self }
    @inlinable static var one: Self { 1 }
}

//==============================================================================
/// pow(x)
/// computes elementwise `x` to the power of `y`
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable public func pow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E.Value: Real
{
    assert(x.shape == y.shape, _messageTensorShapeMismatch)
    var result = Tensor(like: x)
    Context.currentQueue.pow(x, y, &result)
    return result
}

@derivative(of: pow)
@usableFromInline func _vjpPow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError()
    //        let value = pow(x, y)
    //        return (value, { v in
    //            let safeX = x.replacing(with: 1, where: x .<= 0)
    //            let lhsGrad = v * y * pow(x, y - 1)
    //            let rhsGrad = value * v * log(safeX)
    //            return (T(repeating: lhsGrad.sum().element, like: x),
    //                    T(repeating: rhsGrad.sum().element, like: y))
    //        })
}

@inlinable public func pow<S,E>(_ x: Tensor<S,E>, _ n: Int) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.pow(x, n, &result)
    return result
}

// Tensor extension
public extension Tensor where TensorElement.Value: Real {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func pow(_ x: Self, _ y: Self) -> Self { SwiftRTCore.pow(x, y) }
}

//==============================================================================
/// root(x:n:
/// computes the nth root of `x`
/// - Parameter x: value tensor
/// - Parameter n: power
/// - Returns: result
@inlinable public func root<S,E>(_ x: Tensor<S,E>, _ n: Int) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.root(x, n, &result)
    return result
}

@derivative(of: root)
@usableFromInline func _vjpRoot<S,E>(_ x: Tensor<S,E>, _ n: Int)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sqrt<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sqrt(x, &result)
    return result
}

@derivative(of: sqrt)
@usableFromInline func _vjpSqrt<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    let value = sqrt(x)
    return (value, { v in v / (2 * value) })
}

// Tensor extension
public extension Tensor where TensorElement.Value: Real {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sqrt(_ x: Self) -> Self { SwiftRTCore.sqrt(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sqrt() -> Self { sqrt(self) }
}

//==============================================================================
/// sign(x)
///
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable public func sign<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Comparable & SignedNumeric
{
    var result = Tensor(like: x)
    Context.currentQueue.sign(x, &result)
    return result
}

@derivative(of: sign)
@usableFromInline func _vjpSign<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape,
          E.Value: DifferentiableNumeric & Comparable & SignedNumeric
{
    (sign(x), { _ in repeating(0, like: x) })
}

// Tensor extension
public extension Tensor where TensorElement.Value: Comparable & SignedNumeric {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sign(_ x: Self) -> Self { SwiftRTCore.sign(x) }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sign() -> Self { sign(self) }
}

//==============================================================================
/// sigmoid(x)
/// Returns the sigmoid of the specified tensor element-wise.
/// Specifically, computes `1 / (1 + exp(-x))`.
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable public func sigmoid<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sigmoid(x, &result)
    return result
}

@derivative(of: sigmoid)
@usableFromInline func _vjpSigmoid<S,E>(_ x: Tensor<S,E>)
-> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    (sigmoid(x), { v in
        fatalError()
    })
}

// Tensor extension
public extension Tensor where TensorElement.Value: Real {
    // make glboal function visible for extension implementations
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sigmoid(_ x: Self) -> Self { SwiftRTCore.sigmoid(x) }
    
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func sigmoid() -> Self { sign(self) }
}

//==============================================================================
/// tan(x)
/// computes the tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func tan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.tan(x, &result)
    return result
}

@derivative(of: tan)
@usableFromInline func _vjpTan<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    let value = tan(x)
    return (value, { v in v * (1 + value.squared()) })
}

//==============================================================================
/// tanh(x)
/// computes the hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func tanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E.Value: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.tanh(x, &result)
    return result
}

@derivative(of: tanh)
@usableFromInline func _vjpTanh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E.Value: DifferentiableNumeric & Real
{
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
}
