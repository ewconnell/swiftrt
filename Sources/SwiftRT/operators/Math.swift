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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.abs(x, &result)
    return result
}

@derivative(of: abs)
@inlinable func _vjpAbs<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    let signX = sign(x)
    return (abs(x), { $0 * signX })
}

// Tensor extension to disambiguate with Swift.abs
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func abs(_ x: Self) -> Self { SwiftRT.abs(x) }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func abs() -> Self { abs(self) }
}

//==============================================================================
/// acos(x)
/// computes the inverse cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.acos(x, &result)
    return result
}

@derivative(of: acos)
@inlinable func _vjpAcos<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (acos(x), { v in -v / sqrt(1 - x.squared()) })
}

//==============================================================================
/// acosh(x)
/// computes the inverse hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func acosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.acosh(x, &result)
    return result
}

@derivative(of: acosh)
@inlinable func _vjpAcosh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (acosh(x), { v in v / asinh(x) })
}

//==============================================================================
/// asin(x)
/// computes the inverse sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.asin(x, &result)
    return result
}

@derivative(of: asin)
@inlinable func _vjpAsin<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (asin(x), { v in v / sqrt(1 - x.squared()) })
}

//==============================================================================
/// asinh(x)
/// computes the inverse hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func asinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: DifferentiableElement & Real
{
    var result = Tensor(like: x)
    Context.currentQueue.asinh(x, &result)
    return result
}

@derivative(of: asinh)
@inlinable func _vjpAsinh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (asinh(x), { v in v / acosh(x) })
}

//==============================================================================
/// atan(x)
/// computes the inverse tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atan(x, &result)
    return result
}

@derivative(of: atan)
@inlinable func _vjpAtan<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (atan(x), { v in v / (1 + x.squared()) })
}

//==============================================================================
/// atanh(x)
/// computes the inverse hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func atanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atanh(x, &result)
    return result
}

@derivative(of: atanh)
@inlinable func _vjpAtanh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
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
    -> Tensor<S,E> where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.atan2(y, x, &result)
    return result
}

@derivative(of: atan2)
@inlinable func _vjpAtan2<S,E>(y: Tensor<S,E>, x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Real
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
    where S: TensorShape, E: BinaryFloatingPoint, OE: BinaryInteger
{
    var result = Tensor<S,E>(other.shape)
    Context.currentQueue.cast(from: other, to: &result)
    return result
}

@inlinable public func cast<S,E,OE>(_ other: Tensor<S,OE>) -> Tensor<S,E>
    where S: TensorShape, E: BinaryInteger, OE: BinaryFloatingPoint
{
    var result = Tensor<S,E>(other.shape)
    Context.currentQueue.cast(from: other, to: &result)
    return result
}

//==============================================================================
/// cos(x)
/// computes the cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.cos(x, &result)
    return result
}

@derivative(of: cos)
@inlinable func _vjpCos<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (cos(x), { v in -v * sin(x) })
}

//==============================================================================
/// cosh(x)
/// computes the hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func cosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.cosh(x, &result)
    return result
}

@derivative(of: cosh)
@inlinable func _vjpCosh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (cosh(x), { v in v * sinh(x) })
}

//==============================================================================
/// erf(x)
/// computes the error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erf<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.erf(x, &result)
    return result
}

@derivative(of: erf)
@inlinable func _vjpErf<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// erfc(x)
/// computes the complementary error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func erfc<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.erfc(x, &result)
    return result
}

@derivative(of: erfc)
@inlinable func _vjpErfc<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.exp(x, &result)
    return result
}

@derivative(of: exp)
@inlinable func _vjpExp<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    let value = exp(x)
    return (value, { v in value * v } )
}

// Tensor extension
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func exp(_ x: Self) -> Self { SwiftRT.exp(x) }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func exp() -> Self { exp(self) }
}

//==============================================================================
/// exp2(x)
/// Returns two raised to the power of the specified tensor element-wise.
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func exp2<S,E>(_ x: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Real
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
    -> Tensor<S,E> where S: TensorShape, E: Real
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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.expMinusOne(x, &result)
    return result
}

@derivative(of: expMinusOne)
@inlinable func _vjpExpMinusOne<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.gamma(x, &result)
    return result
}

@derivative(of: gamma)
@inlinable func _vjpGamma<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
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
    -> Tensor<S,E> where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.hypot(x, y, &result)
    return result
}

@derivative(of: hypot)
@inlinable func _vjpHypot<S,E>(x: Tensor<S,E>, y: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Real
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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log(x, &result)
    return result
}

@derivative(of: log(_:))
@inlinable func _vjpLog<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (log(x), { v in v / x })
}

@inlinable public func log2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log2(x, &result)
    return result
}

@inlinable public func log10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log10(x, &result)
    return result
}

// Tensor extension
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func log(_ x: Self) -> Self { SwiftRT.log(x) }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func log() -> Self { log(self) }
}

//==============================================================================
/// log(onePlus x:
/// computes one plus the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func log<S,E>(onePlus x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.log(onePlus: x, &result)
    return result
}

@derivative(of: log(onePlus:))
@inlinable func _vjpLogOnePlus<S,E>(onePlus x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// logGamma(x)
/// computes the log gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func logGamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.logGamma(x, &result)
    return result
}

@derivative(of: logGamma)
@inlinable func _vjpLogGamma<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func neg<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: SignedNumeric
{
    var result = Tensor(like: x)
    Context.currentQueue.neg(x, &result)
    return result
}

//@derivative(of: neg)
@inlinable func _vjpNeg<S,E>(_ x: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & SignedNumeric
{
    (-x, { v in -v })
}

// Tensor extension
public extension Tensor where Element: SignedNumeric {
    // make glboal function visible for extension implementations
//    @differentiable(where Element: DifferentiableElement)
    @inlinable static prefix func - (x: Self) -> Self { SwiftRT.neg(x) }

//    @differentiable(where Element: DifferentiableElement)
    @inlinable func neg() -> Self { -self }
}

//==============================================================================
/// sin(x)
/// computes the sign of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sin(x, &result)
    return result
}

@derivative(of: sin)
@inlinable func _vjpSin<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (sin(x), { v in v * cos(x) })
}

//==============================================================================
/// sinh(x)
/// computes the hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sinh(x, &result)
    return result
}

@derivative(of: sinh)
@inlinable func _vjpSinh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (sinh(x), { v in v * cosh(x) })
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func squared<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Numeric
{
    var result = Tensor(like: x)
    Context.currentQueue.squared(x, &result)
    return result
}

@derivative(of: squared)
@inlinable func _vjpSquared<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (squared(x), { v in v * (x + x) })
}

// Tensor extension
public extension Tensor where Element: Numeric {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func squared(_ x: Self) -> Self { SwiftRT.squared(x) }
    
    @differentiable(where Element: DifferentiableElement)
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
    -> Tensor<S,E> where S: TensorShape, E: Real
{
    assert(x.shape == y.shape, _messageTensorExtentsMismatch)
    var result = Tensor(like: x)
    Context.currentQueue.pow(x, y, &result)
    return result
}

@derivative(of: pow)
@inlinable func _vjpPow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Real
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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.pow(x, n, &result)
    return result
}

// Tensor extension
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func pow(_ x: Self, _ y: Self) -> Self { SwiftRT.pow(x, y) }
}

//==============================================================================
/// root(x:n:
/// computes the nth root of `x`
/// - Parameter x: value tensor
/// - Parameter n: power
/// - Returns: result
@inlinable public func root<S,E>(_ x: Tensor<S,E>, _ n: Int) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.root(x, n, &result)
    return result
}

@derivative(of: root)
@inlinable func _vjpRoot<S,E>(_ x: Tensor<S,E>, _ n: Int)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Real
{
    fatalError("Not implemented")
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func sqrt<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sqrt(x, &result)
    return result
}

@derivative(of: sqrt)
@inlinable func _vjpSqrt<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    let value = sqrt(x)
    return (value, { v in v / (2 * value) })
}

// Tensor extension
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func sqrt(_ x: Self) -> Self { SwiftRT.sqrt(x) }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func sqrt() -> Self { sqrt(self) }
}

//==============================================================================
/// sign(x)
///
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable public func sign<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.sign(x, &result)
    return result
}

@derivative(of: sign)
@inlinable func _vjpSign<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    (sign(x), { _ in repeating(0, like: x) })
}

// Tensor extension
public extension Tensor where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Element: DifferentiableElement)
    @inlinable func sign(_ x: Self) -> Self { SwiftRT.sign(x) }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func sign() -> Self { sign(self) }
}

//==============================================================================
/// tan(x)
/// computes the tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func tan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.tan(x, &result)
    return result
}

@derivative(of: tan)
@inlinable func _vjpTan<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
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
    where S: TensorShape, E: Real
{
    var result = Tensor(like: x)
    Context.currentQueue.tanh(x, &result)
    return result
}

@derivative(of: tanh)
@inlinable func _vjpTanh<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
    where S: TensorShape, E: DifferentiableElement & Real
{
    let value = tanh(x)
    return (value, { v in v * (1 - value.squared()) })
}
