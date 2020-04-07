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

////==============================================================================
///// abs(x)
///// computes the absolute value of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func abs<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.abs(x)
//}
//
//@inlinable
//@derivative(of: abs)
//func _vjpAbs<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAbs(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func abs<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.abs(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: abs)
//    func _vjpAbs<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let signX = sign(x)
//        return (abs(x), { $0 * signX })
//    }
//}
//
//// Tensor extension to disambiguate with Swift.abs
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @inlinable
//    @differentiable(where Self: DifferentiableTensorView)
//    func abs(_ x: Self) -> Self { Context.platform.abs(x) }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensorView)
//    func abs() -> Self { abs(self) }
//}
//
////==============================================================================
///// acos(x)
///// computes the inverse cosine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func acos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.acos(x)
//}
//
//@inlinable
//@derivative(of: acos)
//func _vjpAcos<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAcos(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func acos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.acos(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: acos)
//    func _vjpAcos<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (acos(x), { v in -v / self.sqrt(1 - x.squared()) })
//    }
//}
//
////==============================================================================
///// acosh(x)
///// computes the inverse hyperbolic cosine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func acosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.acosh(x)
//}
//
//@inlinable
//@derivative(of: acosh)
//func _vjpAcosh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAcosh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func acosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.acosh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: acosh)
//    func _vjpAcosh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (acosh(x), { v in v / self.asinh(x) })
//    }
//}
//
////==============================================================================
///// asin(x)
///// computes the inverse sine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func asin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.asin(x)
//}
//
//@inlinable
//@derivative(of: asin)
//func _vjpAsin<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAsin(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func asin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.asin(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: asin)
//    func _vjpAsin<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (asin(x), { v in v / self.sqrt(1 - x.squared()) })
//    }
//}
//
////==============================================================================
///// asinh(x)
///// computes the inverse hyperbolic sine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func asinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.asinh(x)
//}
//
//@inlinable
//@derivative(of: asinh)
//func _vjpAsinh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAsinh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func asinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.asinh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: asinh)
//    func _vjpAsinh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (asinh(x), { v in v / self.acosh(x) })
//    }
//}
//
////==============================================================================
///// atan(x)
///// computes the inverse tangent of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func atan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.atan(x)
//}
//
//@inlinable
//@derivative(of: atan)
//func _vjpAtan<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAtan(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func atan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.atan(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: atan)
//    func _vjpAtan<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (atan(x), { v in v / (1 + x.squared()) })
//    }
//}
//
////==============================================================================
///// atanh(x)
///// computes the inverse hyperbolic tangent of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func atanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.atanh(x)
//}
//
//@inlinable
//@derivative(of: atanh)
//func _vjpAtanh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAtanh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func atanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.atanh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: atanh)
//    func _vjpAtanh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (atanh(x), { v in v / (1 - x.squared()) })
//    }
//}
//
////==============================================================================
///// atan2(y:x:
///// computes the arc tangent of a pair of values
///// - Parameter y: value tensor
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func atan2<S,E>(y: T, x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.atan2(y: y, x: x)
//}
//
//@inlinable
//@derivative(of: atan2)
//func _vjpAtan2<S,E>(y: T, x: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpAtan2(y: y, x: x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func atan2<S,E>(y: T, x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.atan2(y: read(y), x: read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: atan2)
//    func _vjpAtan2<S,E>(y: T, x: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//        where T: DifferentiableTensorView, E: Real
//    {
//        // TODO
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// cast(from:to:
///// casts elements of `x` to the output type
///// - Parameter other: value tensor
///// - Returns: result
//@inlinable
//public func cast<T, U>(_ other: U) -> Tensor<S,E> where
//    T: TensorView, E: BinaryFloatingPoint,
//    U: TensorView, U.Element: BinaryInteger, U.Bounds == T.Bounds
//{
//    Context.platform.cast(other)
//}
//
//@inlinable
//public func cast<T, U>(_ other: U) -> Tensor<S,E> where
//    T: TensorView, E: BinaryInteger,
//    U: TensorView, U.Element: BinaryFloatingPoint, U.Bounds == T.Bounds
//{
//    Context.platform.cast(other)
//}
//
//// Platform extension
//public extension Platform {
//    /// cast(other:
//    /// casts from one the other element type to this tensors element type
//    // Integer -> FloatingPoint
//    @inlinable
//    func cast<T, U>(_ other: U) -> Tensor<S,E> where
//        T: TensorView, E: BinaryFloatingPoint,
//        U: TensorView, U.Element: BinaryInteger, U.Bounds == T.Bounds
//    {
//        var result = T.create(other.shape.dense)
//        var resultBuffer = write(&result)
//        currentQueue.cast(from: read(other), to: &resultBuffer)
//        return result
//    }
//
//    // FloatingPoint -> Integer
//    @inlinable
//    func cast<T, U>(_ other: U) -> Tensor<S,E> where
//        T: TensorView, E: BinaryInteger,
//        U: TensorView, U.Element: BinaryFloatingPoint, U.Bounds == T.Bounds
//    {
//        var result = T.create(other.shape.dense)
//        var resultBuffer = write(&result)
//        currentQueue.cast(from: read(other), to: &resultBuffer)
//        return result
//    }
//}
//
////==============================================================================
///// cos(x)
///// computes the cosine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func cos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.cos(x)
//}
//
//@inlinable
//@derivative(of: cos)
//func _vjpCos<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpCos(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func cos<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.cos(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: cos)
//    func _vjpCos<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (cos(x), { v in -v * self.sin(x) })
//    }
//}
//
////==============================================================================
///// cosh(x)
///// computes the hyperbolic cosine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func cosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.cosh(x)
//}
//
//@inlinable
//@derivative(of: cosh)
//func _vjpCosh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpCosh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func cosh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.cosh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: cosh)
//    func _vjpCosh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (cosh(x), { v in v * self.sinh(x) })
//    }
//}
//
////==============================================================================
///// erf(x)
///// computes the error function of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func erf<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.erf(x)
//}
//
//@inlinable
//@derivative(of: erf)
//func _vjpErf<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpErf(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func erf<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.erf(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: erf)
//    func _vjpErf<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// erfc(x)
///// computes the complementary error function of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func erfc<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.erfc(x)
//}
//
//@inlinable
//@derivative(of: erfc)
//func _vjpErfc<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpErfc(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func erfc<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.erfc(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: erfc)
//    func _vjpErfc<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// exp(x)
///// computes the exponential value of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func exp<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.exp(x)
//}
//
//@inlinable
//@derivative(of: exp)
//func _vjpExp<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpExp(x)
//}
//
///// Returns two raised to the power of the specified tensor element-wise.
//@inlinable
//public func exp2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.exp2(x)
//}
//
///// Returns ten raised to the power of the specified tensor element-wise.
//@inlinable
//public func exp10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.exp10(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func exp<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.exp(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: exp)
//    func _vjpExp<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let value = exp(x)
//        return (value, { v in value * v } )
//    }
//
//    /// Returns two raised to the power of the specified tensor element-wise.
//    @inlinable
//    func exp2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.exp2(read(x), &resultBuffer)
//        return result
//    }
//
//    /// Returns ten raised to the power of the specified tensor element-wise.
//    @inlinable
//    func exp10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.exp10(read(x), &resultBuffer)
//        return result
//    }
//}
//
//// Tensor extension
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func exp(_ x: Self) -> Self { Context.platform.exp(x) }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func exp() -> Self { exp(self) }
//}
//
////==============================================================================
///// expMinusOne(x)
///// computes the exponential minus one value of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func expMinusOne<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.expMinusOne(x)
//}
//
//@inlinable
//@derivative(of: expMinusOne)
//func _vjpExpMinusOne<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpExpMinusOne(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func expMinusOne<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.expMinusOne(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: expMinusOne)
//    func _vjpExpMinusOne<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let y = expMinusOne(x)
//        return (y, { v in v * y })
//    }
//}
//
////==============================================================================
///// gamma(x)
///// computes the gamma of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func gamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.gamma(x)
//}
//
//@inlinable
//@derivative(of: gamma)
//func _vjpGamma<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpGamma(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func gamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.gamma(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: gamma)
//    func _vjpGamma<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// hypot(x:y:
///// calculate the length of the hypotenuse of a right triangle
///// - Parameter x: value tensor
///// - Parameter y: value tensor
///// - Returns: result
//@inlinable
//public func hypot<S,E>(_ x: T, _ y: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.hypot(x, y)
//}
//
//@inlinable
//@derivative(of: hypot)
//func _vjpHypot<S,E>(x: T, y: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpHypot(x, y)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func hypot<S,E>(_ x: T, _ y: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.hypot(read(x), read(y), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: hypot)
//    func _vjpHypot<S,E>(_ x: T, _ y: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//        where T: DifferentiableTensorView, E: Real
//    {
//        // TODO
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// log(x)
///// computes the log of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func log<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.log(x)
//}
//
//@inlinable
//@derivative(of: log(_:))
//func _vjpLog<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpLog(x)
//}
//
//@inlinable
//public func log2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.log2(x)
//}
//
//@inlinable
//public func log10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//    Context.platform.log10(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func log<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.log(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: log(_:))
//    func _vjpLog<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (log(x), { v in v / x })
//    }
//
//    @inlinable
//    func log2<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.log2(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    func log10<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E> where S: TensorShape, E: Real {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.log10(read(x), &resultBuffer)
//        return result
//    }
//}
//
//// Tensor extension
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func log(_ x: Self) -> Self { Context.platform.log(x) }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func log() -> Self { log(self) }
//}
//
////==============================================================================
///// log(onePlus x:
///// computes one plus the log of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func log<S,E>(onePlus x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.log(onePlus: x)
//}
//
//@inlinable
//@derivative(of: log(onePlus:))
//func _vjpLogOnePlus<S,E>(onePlus x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpLogOnePlus(onePlus: x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func log<S,E>(onePlus x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.log(onePlus: read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: log(onePlus:))
//    func _vjpLogOnePlus<S,E>(onePlus x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// logGamma(x)
///// computes the log gamma of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func logGamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.logGamma(x)
//}
//
//@inlinable
//@derivative(of: logGamma)
//func _vjpLogGamma<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpLogGamma(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func logGamma<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.logGamma(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: logGamma)
//    func _vjpLogGamma<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
//==============================================================================
/// neg(x)
/// computes the negated value of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func neg<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: SignedNumeric
{
    var result = empty(like: x)
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
//    @differentiable(where Self: DifferentiableTensorView)
    @inlinable static prefix func - (x: Self) -> Self { SwiftRT.neg(x) }

//    @differentiable(where Self: DifferentiableTensorView)
    @inlinable func neg() -> Self { -self }
}

////==============================================================================
///// sin(x)
///// computes the sign of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func sin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.sin(x)
//}
//
//@inlinable
//@derivative(of: sin)
//func _vjpSin<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpSin(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func sin<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.sin(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: sin)
//    func _vjpSin<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (sin(x), { v in v * self.cos(x) })
//    }
//}
//
////==============================================================================
///// sinh(x)
///// computes the hyperbolic sine of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func sinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.sinh(x)
//}
//
//@inlinable
//@derivative(of: sinh)
//func _vjpSinh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpSinh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func sinh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.sinh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: sinh)
//    func _vjpSinh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (sinh(x), { v in v * self.cosh(x) })
//    }
//}

//==============================================================================
/// squared(x)
/// computes the elementwise squares of `x`
/// - Parameter x: value tensor
/// - Returns: result
@inlinable public func squared<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Numeric
{
    var result = empty(like: x)
    Context.currentQueue.squared(x, &result)
    return result
}

//@derivative(of: squared)
@inlinable func _vjpSquared<S,E>(_ x: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (squared(x), { v in v * (x + x) })
}

// Tensor extension
public extension Tensor where Element: Numeric {
    // make glboal function visible for extension implementations
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable func squared(_ x: Self) -> Self { SwiftRT.squared(x) }
    
//    @differentiable(where Self: DifferentiableTensorView)
    @inlinable func squared() -> Self { squared(self) }
}

/// Numeric extension for scalar types
public extension Numeric {
    @inlinable func squared() -> Self { self * self }
    @inlinable static var one: Self { 1 }
}
//
////==============================================================================
///// pow(x)
///// computes elementwise `x` to the power of `y`
///// - Parameter x: value tensor
///// - Parameter y: power tensor
///// - Returns: result
//@inlinable
//public func pow<S,E>(_ x: T, _ y: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.pow(x, y)
//}
//
//@inlinable
//@derivative(of: pow)
//func _vjpPow<S,E>(_ x: T, _ y: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpPow(x, y)
//}
//
//@inlinable
//public func pow<S,E>(_ x: T, _ n: Int) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.pow(x, n)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func pow<S,E>(_ x: T, _ y: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        assert(x.bounds == y.bounds, _messageTensorExtentsMismatch)
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.pow(read(x), read(y), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: pow)
//    func _vjpPow<S,E>(_ x: T, _ y: Tensor<S,E>) -> (value: T, pullback: (T) -> (T, T))
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError()
//        //        let value = pow(x, y)
//        //        return (value, { v in
//        //            let safeX = x.replacing(with: 1, where: x .<= 0)
//        //            let lhsGrad = v * y * pow(x, y - 1)
//        //            let rhsGrad = value * v * log(safeX)
//        //            return (T(repeating: lhsGrad.sum().element, like: x),
//        //                    T(repeating: rhsGrad.sum().element, like: y))
//        //        })
//    }
//
//    @inlinable
//    func pow<S,E>(_ x: T, _ n: Int) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.pow(read(x), n, &resultBuffer)
//        return result
//    }
//}
//
//// Tensor extension
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @inlinable
//    @differentiable(where Self: DifferentiableTensorView)
//    func pow(_ x: Self, _ y: Self) -> Self { Context.platform.pow(x, y) }
//}
//
////==============================================================================
///// root(x:n:
///// computes the nth root of `x`
///// - Parameter x: value tensor
///// - Parameter n: power
///// - Returns: result
//@inlinable
//public func root<S,E>(_ x: T, _ n: Int) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.root(x, n)
//}
//
//@inlinable
//@derivative(of: root)
//func _vjpRoot<S,E>(_ x: T, _ n: Int) -> (value: T, pullback: (T) -> (T))
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpRoot(x, n)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func root<S,E>(_ x: T, _ n: Int) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.root(read(x), n, &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: root)
//    func _vjpRoot<S,E>(_ x: T, _ n: Int) -> (value: T, pullback: (T) -> (T))
//        where T: DifferentiableTensorView, E: Real
//    {
//        fatalError("Not implemented")
//    }
//}
//
////==============================================================================
///// sqrt(x)
///// computes the square root of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func sqrt<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.sqrt(x)
//}
//
//@inlinable
//@derivative(of: sqrt)
//func _vjpSqrt<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpSqrt(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func sqrt<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.sqrt(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: sqrt)
//    func _vjpSqrt<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let value = sqrt(x)
//        return (value, { v in v / (2 * value) })
//    }
//}
//
//// Tensor extension
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sqrt(_ x: Self) -> Self { Context.platform.sqrt(x) }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sqrt() -> Self { sqrt(self) }
//}
//
////==============================================================================
///// sign(x)
/////
///// - Parameter x: value tensor
///// - Returns: the signs of `x`. -1 for negative `x` values, 1 for positive
//@inlinable
//public func sign<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.sign(x)
//}
//
//@inlinable
//@derivative(of: sign)
//func _vjpSign<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpSign(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func sign<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.sign(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: sign)
//    func _vjpSign<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        (sign(x), { _ in T(repeating: 0, like: x) })
//    }
//}
//
//// Tensor extension
//public extension TensorView where Element: Real {
//    // make glboal function visible for extension implementations
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sign(_ x: Self) -> Self { Context.platform.sign(x) }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sign() -> Self { sign(self) }
//}
//
////==============================================================================
///// tan(x)
///// computes the tangent of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func tan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.tan(x)
//}
//
//@inlinable
//@derivative(of: tan)
//func _vjpTan<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpTan(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func tan<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.tan(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: tan)
//    func _vjpTan<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let value = tan(x)
//        return (value, { v in v * (1 + value.squared()) })
//    }
//}
//
////==============================================================================
///// tanh(x)
///// computes the hyperbolic tangent of `x`
///// - Parameter x: value tensor
///// - Returns: result
//@inlinable
//public func tanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//    where S: TensorShape, E: Real
//{
//    Context.platform.tanh(x)
//}
//
//@inlinable
//@derivative(of: tanh)
//func _vjpTanh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//    where T: DifferentiableTensorView, E: Real
//{
//    Context.platform._vjpTanh(x)
//}
//
//// Platform extension
//public extension Platform {
//    @inlinable
//    func tanh<S,E>(_ x: Tensor<S,E>) -> Tensor<S,E>
//        where S: TensorShape, E: Real
//    {
//        var (result, resultBuffer) = createResult(like: x)
//        currentQueue.tanh(read(x), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: tanh)
//    func _vjpTanh<S,E>(_ x: Tensor<S,E>) -> (value: T, pullback: (T) -> Tensor<S,E>)
//        where T: DifferentiableTensorView, E: Real
//    {
//        let value = tanh(x)
//        return (value, { v in v * (1 - value.squared()) })
//    }
//}
