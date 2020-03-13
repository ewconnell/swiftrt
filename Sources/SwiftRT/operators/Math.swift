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
@inlinable
public func abs<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.abs(x)
}

@inlinable
@derivative(of: abs)
func _vjpAbs<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAbs(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func abs<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.abs(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: abs)
    func _vjpAbs<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let signX = sign(x)
        return (abs(x), { $0 * signX })
    }
}

// Tensor extension to disambiguate with Swift.abs
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    func abs(_ x: Self) -> Self { Platform.service.abs(x) }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    func abs() -> Self { abs(self) }
}

//==============================================================================
/// acos(x)
/// computes the inverse cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func acos<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.acos(x)
}

@inlinable
@derivative(of: acos)
func _vjpAcos<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAcos(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func acos<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.acos(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: acos)
    func _vjpAcos<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (acos(x), { v in -v / self.sqrt(1 - x.squared()) })
    }
}

//==============================================================================
/// acosh(x)
/// computes the inverse hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func acosh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.acosh(x)
}

@inlinable
@derivative(of: acosh)
func _vjpAcosh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAcosh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func acosh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.acosh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: acosh)
    func _vjpAcosh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (acosh(x), { v in v / self.asinh(x) })
    }
}

//==============================================================================
/// asin(x)
/// computes the inverse sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func asin<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.asin(x)
}

@inlinable
@derivative(of: asin)
func _vjpAsin<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAsin(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func asin<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.asin(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: asin)
    func _vjpAsin<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (asin(x), { v in v / self.sqrt(1 - x.squared()) })
    }
}

//==============================================================================
/// asinh(x)
/// computes the inverse hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func asinh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.asinh(x)
}

@inlinable
@derivative(of: asinh)
func _vjpAsinh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAsinh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func asinh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.asinh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: asinh)
    func _vjpAsinh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (asinh(x), { v in v / self.acosh(x) })
    }
}

//==============================================================================
/// atan(x)
/// computes the inverse tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func atan<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.atan(x)
}

@inlinable
@derivative(of: atan)
func _vjpAtan<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAtan(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func atan<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.atan(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: atan)
    func _vjpAtan<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (atan(x), { v in v / (1 + x.squared()) })
    }
}

//==============================================================================
/// atanh(x)
/// computes the inverse hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func atanh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.atanh(x)
}

@inlinable
@derivative(of: atanh)
func _vjpAtanh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAtanh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func atanh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.atanh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: atanh)
    func _vjpAtanh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (atanh(x), { v in v / (1 - x.squared()) })
    }
}

//==============================================================================
/// atan2(y:x:
/// computes the arc tangent of a pair of values
/// - Parameter y: value tensor
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func atan2<T>(y: T, x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.atan2(y: y, x: x)
}

@inlinable
@derivative(of: atan2)
func _vjpAtan2<T>(y: T, x: T) -> (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpAtan2(y: y, x: x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func atan2<T>(y: T, x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.atan2(y: read(y), x: read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: atan2)
    func _vjpAtan2<T>(y: T, x: T) -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Real
    {
        // TODO
        fatalError("Not implemented")
    }
}

//==============================================================================
/// cast(from:to:
/// casts elements of `x` to the output type
/// - Parameter other: value tensor
/// - Returns: result
@inlinable
public func cast<T, U>(_ other: U) -> T where
    T: TensorView, T.Element: BinaryFloatingPoint,
    U: TensorView, U.Element: BinaryInteger, U.Bounds == T.Bounds
{
    Platform.service.cast(other)
}

@inlinable
public func cast<T, U>(_ other: U) -> T where
    T: TensorView, T.Element: BinaryInteger,
    U: TensorView, U.Element: BinaryFloatingPoint, U.Bounds == T.Bounds
{
    Platform.service.cast(other)
}

// Platform extension
public extension PlatformService {
    /// cast(other:
    /// casts from one the other element type to this tensors element type
    // Integer -> FloatingPoint
    @inlinable
    func cast<T, U>(_ other: U) -> T where
        T: TensorView, T.Element: BinaryFloatingPoint,
        U: TensorView, U.Element: BinaryInteger, U.Bounds == T.Bounds
    {
        var result = T.create(other.shape.dense, nil)
        var resultBuffer = write(&result)
        currentQueue.cast(from: read(other), to: &resultBuffer)
        return result
    }

    // FloatingPoint -> Integer
    @inlinable
    func cast<T, U>(_ other: U) -> T where
        T: TensorView, T.Element: BinaryInteger,
        U: TensorView, U.Element: BinaryFloatingPoint, U.Bounds == T.Bounds
    {
        var result = T.create(other.shape.dense, nil)
        var resultBuffer = write(&result)
        currentQueue.cast(from: read(other), to: &resultBuffer)
        return result
    }
}

//==============================================================================
/// cos(x)
/// computes the cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func cos<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.cos(x)
}

@inlinable
@derivative(of: cos)
func _vjpCos<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpCos(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func cos<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.cos(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: cos)
    func _vjpCos<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (cos(x), { v in -v * self.sin(x) })
    }
}

//==============================================================================
/// cosh(x)
/// computes the hyperbolic cosine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func cosh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.cosh(x)
}

@inlinable
@derivative(of: cosh)
func _vjpCosh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpCosh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func cosh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.cosh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: cosh)
    func _vjpCosh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (cosh(x), { v in v * self.sinh(x) })
    }
}

//==============================================================================
/// erf(x)
/// computes the error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func erf<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.erf(x)
}

@inlinable
@derivative(of: erf)
func _vjpErf<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpErf(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func erf<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.erf(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: erf)
    func _vjpErf<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// erfc(x)
/// computes the complementary error function of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func erfc<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.erfc(x)
}

@inlinable
@derivative(of: erfc)
func _vjpErfc<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpErfc(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func erfc<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.erfc(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: erfc)
    func _vjpErfc<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// exp(x)
/// computes the exponential value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func exp<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.exp(x)
}

@inlinable
@derivative(of: exp)
func _vjpExp<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpExp(x)
}

/// Returns two raised to the power of the specified tensor element-wise.
@inlinable
public func exp2<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.exp2(x)
}

/// Returns ten raised to the power of the specified tensor element-wise.
@inlinable
public func exp10<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.exp10(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func exp<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.exp(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: exp)
    func _vjpExp<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let value = exp(x)
        return (value, { v in value * v } )
    }
    
    /// Returns two raised to the power of the specified tensor element-wise.
    @inlinable
    func exp2<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.exp2(read(x), &resultBuffer)
        return result
    }

    /// Returns ten raised to the power of the specified tensor element-wise.
    @inlinable
    func exp10<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.exp10(read(x), &resultBuffer)
        return result
    }
}

// Tensor extension
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func exp(_ x: Self) -> Self { Platform.service.exp(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func exp() -> Self { exp(self) }
}

//==============================================================================
/// expMinusOne(x)
/// computes the exponential minus one value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func expMinusOne<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.expMinusOne(x)
}

@inlinable
@derivative(of: expMinusOne)
func _vjpExpMinusOne<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpExpMinusOne(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func expMinusOne<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.expMinusOne(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: expMinusOne)
    func _vjpExpMinusOne<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let y = expMinusOne(x)
        return (y, { v in v * y })
    }
}

//==============================================================================
/// gamma(x)
/// computes the gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func gamma<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.gamma(x)
}

@inlinable
@derivative(of: gamma)
func _vjpGamma<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpGamma(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func gamma<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.gamma(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: gamma)
    func _vjpGamma<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// hypot(x:y:
/// calculate the length of the hypotenuse of a right triangle
/// - Parameter x: value tensor
/// - Parameter y: value tensor
/// - Returns: result
@inlinable
public func hypot<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.hypot(x, y)
}

@inlinable
@derivative(of: hypot)
func _vjpHypot<T>(x: T, y: T) -> (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpHypot(x, y)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func hypot<T>(_ x: T, _ y: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.hypot(read(x), read(y), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: hypot)
    func _vjpHypot<T>(_ x: T, _ y: T) -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Real
    {
        // TODO
        fatalError("Not implemented")
    }
}

//==============================================================================
/// log(x)
/// computes the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func log<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.log(x)
}

@inlinable
@derivative(of: log(_:))
func _vjpLog<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpLog(x)
}

@inlinable
public func log2<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.log2(x)
}

@inlinable
public func log10<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
    Platform.service.log10(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func log<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.log(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: log(_:))
    func _vjpLog<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (log(x), { v in v / x })
    }

    @inlinable
    func log2<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.log2(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    func log10<T>(_ x: T) -> T where T: TensorView, T.Element: Real {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.log10(read(x), &resultBuffer)
        return result
    }
}

// Tensor extension
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func log(_ x: Self) -> Self { Platform.service.log(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func log() -> Self { log(self) }
}

//==============================================================================
/// log(onePlus x:
/// computes one plus the log of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func log<T>(onePlus x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.log(onePlus: x)
}

@inlinable
@derivative(of: log(onePlus:))
func _vjpLogOnePlus<T>(onePlus x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpLogOnePlus(onePlus: x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func log<T>(onePlus x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.log(onePlus: read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: log(onePlus:))
    func _vjpLogOnePlus<T>(onePlus x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// logGamma(x)
/// computes the log gamma of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func logGamma<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.logGamma(x)
}

@inlinable
@derivative(of: logGamma)
func _vjpLogGamma<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpLogGamma(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func logGamma<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.logGamma(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: logGamma)
    func _vjpLogGamma<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func neg<T>(_ x: T) -> T
    where T: TensorView, T.Element: SignedNumeric
{
    Platform.service.neg(x)
}

@inlinable
@derivative(of: neg)
func _vjpNeg<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: SignedNumeric
{
    Platform.service._vjpNeg(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func neg<T>(_ x: T) -> T where T: TensorView, T.Element: SignedNumeric {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.neg(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: neg)
    func _vjpNeg<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: SignedNumeric
    {
        (-x, { v in -v })
    }
}

// Tensor extension
public extension TensorView where Element: SignedNumeric {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    static prefix func - (x: Self) -> Self { Platform.service.neg(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func neg() -> Self { -self }
}

//==============================================================================
/// sin(x)
/// computes the sign of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func sin<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.sin(x)
}

@inlinable
@derivative(of: sin)
func _vjpSin<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpSin(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func sin<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.sin(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: sin)
    func _vjpSin<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (sin(x), { v in v * self.cos(x) })
    }
}

//==============================================================================
/// sinh(x)
/// computes the hyperbolic sine of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func sinh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.sinh(x)
}

@inlinable
@derivative(of: sinh)
func _vjpSinh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpSinh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func sinh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.sinh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: sinh)
    func _vjpSinh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (sinh(x), { v in v * self.cosh(x) })
    }
}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func squared<T>(_ x: T) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.squared(x)
}

@inlinable
@derivative(of: squared)
func _vjpSquared<T>(_ x: T) -> (value: T, pullback: (T) -> (T))
    where T: DifferentiableTensorView
{
    Platform.service._vjpSquared(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func squared<T>(_ x: T) -> T
        where T: TensorView, T.Element: Numeric
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.squared(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: squared)
    func _vjpSquared<T>(_ x: T) -> (value: T, pullback: (T) -> (T))
        where T: DifferentiableTensorView
    {
        (squared(x), { v in v * (x + x) })
    }
}

// Tensor extension
public extension TensorView where Element: Numeric {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func squared(_ x: Self) -> Self { Platform.service.squared(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func squared() -> Self { squared(self) }
}

/// Numeric extension for scalar types
public extension Numeric {
    @inlinable
    func squared() -> Self { self * self }
}

//==============================================================================
/// pow(x)
/// computes elementwise `x` to the power of `y`
/// - Parameter x: value tensor
/// - Parameter y: power tensor
/// - Returns: result
@inlinable
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.pow(x, y)
}

@inlinable
@derivative(of: pow)
func _vjpPow<T>(_ x: T, _ y: T) -> (value: T, pullback: (T) -> (T, T))
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpPow(x, y)
}

@inlinable
public func pow<T>(_ x: T, _ n: Int) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.pow(x, n)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func pow<T>(_ x: T, _ y: T) -> T
        where T: TensorView, T.Element: Real
    {
        assert(x.bounds == y.bounds, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.pow(read(x), read(y), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: pow)
    func _vjpPow<T>(_ x: T, _ y: T) -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Real
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

    @inlinable
    func pow<T>(_ x: T, _ n: Int) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.pow(read(x), n, &resultBuffer)
        return result
    }
}

// Tensor extension
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    func pow(_ x: Self, _ y: Self) -> Self { Platform.service.pow(x, y) }
}

//==============================================================================
/// root(x:n:
/// computes the nth root of `x`
/// - Parameter x: value tensor
/// - Parameter n: power
/// - Returns: result
@inlinable
public func root<T>(_ x: T, _ n: Int) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.root(x, n)
}

@inlinable
@derivative(of: root)
func _vjpRoot<T>(_ x: T, _ n: Int) -> (value: T, pullback: (T) -> (T))
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpRoot(x, n)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func root<T>(_ x: T, _ n: Int) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.root(read(x), n, &resultBuffer)
        return result
    }

    @inlinable
    @derivative(of: root)
    func _vjpRoot<T>(_ x: T, _ n: Int) -> (value: T, pullback: (T) -> (T))
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError("Not implemented")
    }
}

//==============================================================================
/// sqrt(x)
/// computes the square root of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func sqrt<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.sqrt(x)
}

@inlinable
@derivative(of: sqrt)
func _vjpSqrt<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpSqrt(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func sqrt<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.sqrt(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: sqrt)
    func _vjpSqrt<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let value = sqrt(x)
        return (value, { v in v / (2 * value) })
    }
}

// Tensor extension
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrt(_ x: Self) -> Self { Platform.service.sqrt(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrt() -> Self { sqrt(self) }
}

//==============================================================================
/// sign(x)
///
/// - Parameter x: value tensor
/// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
@inlinable
public func sign<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.sign(x)
}

@inlinable
@derivative(of: sign)
func _vjpSign<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpSign(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func sign<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.sign(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: sign)
    func _vjpSign<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        (sign(x), { _ in T(repeating: 0, like: x) })
    }
}

// Tensor extension
public extension TensorView where Element: Real {
    // make glboal function visible for extension implementations
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sign(_ x: Self) -> Self { Platform.service.sign(x) }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sign() -> Self { sign(self) }
}

//==============================================================================
/// tan(x)
/// computes the tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func tan<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.tan(x)
}

@inlinable
@derivative(of: tan)
func _vjpTan<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpTan(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func tan<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.tan(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: tan)
    func _vjpTan<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let value = tan(x)
        return (value, { v in v * (1 + value.squared()) })
    }
}

//==============================================================================
/// tanh(x)
/// computes the hyperbolic tangent of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable
public func tanh<T>(_ x: T) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.tanh(x)
}

@inlinable
@derivative(of: tanh)
func _vjpTanh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
    where T: DifferentiableTensorView, T.Element: Real
{
    Platform.service._vjpTanh(x)
}

// Platform extension
public extension PlatformService {
    @inlinable
    func tanh<T>(_ x: T) -> T
        where T: TensorView, T.Element: Real
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.tanh(read(x), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: tanh)
    func _vjpTanh<T>(_ x: T) -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        let value = tanh(x)
        return (value, { v in v * (1 - value.squared()) })
    }
}
