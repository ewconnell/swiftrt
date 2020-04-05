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
import Numerics

//==============================================================================
/// add
/// performs an elementwise add
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable public func add<T, U>(_ lhs: T, _ rhs: U)
    -> DenseTensor<T.Shape, T.Element> where
    T: Tensor, T.Element: AdditiveArithmetic,
    U: Tensor, U.Element == T.Element
{
    var result = empty(like: lhs)
    Context.currentQueue.add(lhs, rhs, &result)
    return result
}

//@differentiable(where T: DifferentiableTensor)
@inlinable public func add<T>(_ lhs: T, _ rhs: T.Element)
    -> DenseTensor<T.Shape, T.Element>
    where T: Tensor, T.Element: AdditiveArithmetic
{
    add(lhs, repeating(rhs, like: lhs))
}

//@differentiable(where T: DifferentiableTensor)
@inlinable public func add<T>(_ lhs: T.Element, _ rhs: T)
    -> DenseTensor<T.Shape, T.Element>
    where T: Tensor, T.Element: AdditiveArithmetic
{
    add(repeating(lhs, like: rhs), rhs)
}

////@derivative(of: add)
//@inlinable public func _vjpAdd<T, U>(_ lhs: T, _ rhs: U)
//    -> (value: T, pullback: (T) ->(T, T))
//    where T: DifferentiableTensor
//{
//    (lhs + rhs, { v in (v, v) })
//}

public extension Tensor where Element: AdditiveArithmetic {
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +<U>(lhs: Self, rhs: U)
        -> DenseTensor<Shape, Element>
        where U: Tensor, U.Element == Element
    {
        add(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +(lhs: Self, rhs: Element) -> DenseTensor<Shape, Element> {
        add(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +(lhs: Element, rhs: Self) -> DenseTensor<Shape, Element> {
        add(lhs, rhs)
    }

    // VectorProtocol
    
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable func adding(_ x: Element) -> DenseTensor<Shape, Element> {
        self + x
    }
}

////==============================================================================
///// subtract
///// peforms an elementwise subtract
///// - Parameter lhs: left hand tensor
///// - Parameter rhs: right hand tensor
///// - Returns: result
//@inlinable
//public func subtract<T>(_ lhs: T, _ rhs: T) -> T
//    where T: Tensor, T.Element: AdditiveArithmetic
//{
//    Context.platform.subtract(lhs, rhs)
//}
//
//@derivative(of: subtract)
//@inlinable
//public func _vjpSubtract<T>(lhs: T, rhs: T) ->
//    (value: T, pullback: (T) ->(T, T)) where T: DifferentiableTensor
//{
//    Context.platform._vjpSubtract(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func subtract<T>(_ lhs: T, _ rhs: T) -> T
//        where T: Tensor, T.Element: AdditiveArithmetic
//    {
//        let (left, right) = implicitlyMatchExtents(lhs, rhs)
//        assert(left.bounds == right.bounds, _messageTensorExtentsMismatch)
//        var (result, resultBuffer) = createResult(like: left)
//        currentQueue.subtract(read(left), read(right), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: subtract)
//    func _vjpSubtract<T>(_ lhs: T, _ rhs: T) ->
//        (value: T, pullback: (T) ->(T, T))
//        where T: DifferentiableTensor
//    {
//        (lhs - rhs, { v in (v, T.zero - v) })
//    }
//}
//
//public extension Tensor where Element: AdditiveArithmetic {
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func - (lhs: Self, rhs: Self) -> Self { subtract(lhs, rhs) }
//
//    @inlinable
//    static func -= (lhs: inout Self, rhs: Element) {
//        lhs = subtract(lhs, Self(repeating: rhs, like: lhs))
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func - (lhs: Self, rhs: Element) -> Self {
//        subtract(lhs, Self(repeating: rhs, to: lhs.bounds))
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func - (lhs: Element, rhs: Self) -> Self {
//        subtract(Self(repeating: lhs, to: rhs.bounds), rhs)
//    }
//
//    // VectorProtocol
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    func subtracting(_ x: Element) -> Self {
//        self - x
//    }
//}
//
////==============================================================================
///// mul
///// performs an elementwise multiply
///// - Parameter lhs: left hand tensor
///// - Parameter rhs: right hand tensor.
///// - Returns: a new tensor containing the result
//@inlinable
//public func mul<T>(_ lhs: T, _ rhs: T) -> T
//    where T: Tensor, T.Element: Numeric
//{
//    Context.platform.mul(lhs, rhs)
//}
//
//@inlinable
//@derivative(of: mul)
//func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
//    (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensor
//{
//    Context.platform._vjpMultiply(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func mul<T>(_ lhs: T, _ rhs: T) -> T
//        where T: Tensor, T.Element: Numeric
//    {
//        let (left, right) = implicitlyMatchExtents(lhs, rhs)
//        assert(left.bounds == right.bounds, _messageTensorExtentsMismatch)
//        var (result, resultBuffer) = createResult(like: left)
//        currentQueue.mul(read(left), read(right), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: mul)
//    func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
//        (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensor
//    {
//        (lhs * rhs, { v in (v * rhs, v * lhs) })
//    }
//}
//
//public extension Tensor where Element: Numeric {
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }
//
//    @inlinable
//    static func *= (lhs: inout Self, rhs: Element) {
//        lhs = mul(lhs, Self(repeating: rhs, to: lhs.bounds))
//    }
//
//    @inlinable
//    static func *= (lhs: inout Self, rhs: Self) {
//        lhs = lhs * rhs
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func * (lhs: Self, rhs: Element) -> Self {
//        mul(lhs, Self(repeating: rhs, to: lhs.bounds))
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func * (lhs: Element, rhs: Self) -> Self {
//        mul(Self(repeating: lhs, to: rhs.bounds), rhs)
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    func scaled(by scalar: Element) -> Self {
//        self * scalar
//    }
//
//    // TODO: this syntax is incorrect and is only here to conform to
//    // PointwiseMultiplicative and should be removed
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func .* (lhs: Self, rhs: Self) -> Self {
//        lhs * rhs
//    }
//}
//
////==============================================================================
///// div
///// performs an elementwise divide
///// - Parameter lhs: left hand tensor
///// - Parameter rhs: right hand tensor.
///// - Returns: a new tensor containing the result
//@inlinable
//public func div<T>(_ lhs: T, _ rhs: T) -> T
//    where T: Tensor, T.Element: AlgebraicField
//{
//    Context.platform.div(lhs, rhs)
//}
//
//@inlinable
//@derivative(of: div)
//func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
//    (value: T, pullback: (T) -> (T, T)) where
//    T: DifferentiableTensor, T.Element: AlgebraicField
//{
//    Context.platform._vjpDivide(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func div<T>(_ lhs: T, _ rhs: T) -> T
//        where T: Tensor, T.Element: AlgebraicField
//    {
//        let (left, right) = implicitlyMatchExtents(lhs, rhs)
//        assert(left.bounds == right.bounds, _messageTensorExtentsMismatch)
//        var (result, resultBuffer) = createResult(like: left)
//        currentQueue.div(read(left), read(right), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    @derivative(of: div)
//    func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
//        (value: T, pullback: (T) -> (T, T)) where
//        T: DifferentiableTensor, T.Element: AlgebraicField
//    {
//        (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
//    }
//}
//
//public extension Tensor where Element: AlgebraicField {
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }
//
//    @inlinable
//    static func /= (lhs: inout Self, rhs: Element) {
//        lhs = div(lhs, Self(repeating: rhs, to: lhs.bounds))
//    }
//
//    @inlinable
//    static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func / (lhs: Self, rhs: Element) -> Self {
//        div(lhs, Self(repeating: rhs, to: lhs.bounds))
//    }
//
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    static func / (lhs: Element, rhs: Self) -> Self {
//        div(Self(repeating: lhs, to: rhs.bounds), rhs)
//    }
//
//    // PointwiseMultiplicative
//    @inlinable
//    @differentiable(where Self: DifferentiableTensor)
//    var reciprocal: Self {
//        1 / self
//    }
//}
