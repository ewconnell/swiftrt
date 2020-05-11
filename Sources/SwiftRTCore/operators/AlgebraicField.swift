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
/// - Parameters:
///  - lhs: left hand tensor
///  - rhs: right hand tensor
/// - Returns: result
@differentiable(where E: DifferentiableElement)
@inlinable public func add<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E: AdditiveArithmetic
{
    /// REMOVE THIS
    let (lhs, rhs) = match(lhs, rhs)
    assert(lhs.shape == rhs.shape)

    var result = Tensor(like: lhs)
    Context.currentQueue.add(lhs, rhs, &result)
    return result
}

@derivative(of: add)
@inlinable public func _vjpAdd<S, E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    return (lhs + rhs, { v in (v, v) })
}

//------------------------------------------------------------------------------
public extension Tensor where Element: AdditiveArithmetic {
    @differentiable(where Element: DifferentiableElement)
    @inlinable static func +(lhs: Self, rhs: Self) -> Self {
        add(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable static func +(lhs: Self, rhs: Element) -> Self {
        add(lhs, repeating(rhs, like: lhs))
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable static func +=(lhs: inout Self, rhs: Element) {
        lhs = (lhs + rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: rhs where Element: DifferentiableElement)
    @inlinable static func +(lhs: Element, rhs: Self) -> Self {
        add(repeating(lhs, like: rhs), rhs)
    }

    @derivative(of: +, wrt: lhs)
    @inlinable static func _vjpAdd(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs + rhs, { $0 })
    }

    @derivative(of: +, wrt: rhs)
    @inlinable static func _vjpAdd(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs + rhs, { $0 })
    }

    // VectorProtocol
    @differentiable(where Element: DifferentiableElement)
    @inlinable func adding(_ x: Element) -> Self {
        self + x
    }
}

//==============================================================================
/// subtract
/// performs an elementwise subtract
/// - Parameters:
///  - lhs: left hand tensor
///  - rhs: right hand tensor
/// - Returns: result
@differentiable(where E: DifferentiableElement)
@inlinable public func subtract<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E: AdditiveArithmetic
{
    /// REMOVE THIS
    let (lhs, rhs) = match(lhs, rhs)
    assert(lhs.shape == rhs.shape)

    var result = Tensor(like: lhs)
    Context.currentQueue.subtract(lhs, rhs, &result)
    return result
}

@derivative(of: subtract)
@inlinable public func _vjpSubtract<S, E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (lhs - rhs, { v in (v, E.zero - v) })
}

public extension Tensor where Element: AdditiveArithmetic {
    @differentiable(where Element: DifferentiableElement)
    @inlinable static func -(lhs: Self, rhs: Self) -> Self {
        subtract(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable static func -(lhs: Self, rhs: Element) -> Self {
        subtract(lhs, repeating(rhs, like: lhs))
    }

    @derivative(of: -, wrt: lhs)
    @inlinable static func _vjpSubtract(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs - rhs, { $0 })
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: rhs where Element: DifferentiableElement)
    @inlinable static func -(lhs: Element, rhs: Self) -> Self {
        subtract(repeating(lhs, like: rhs), rhs)
    }

    @derivative(of: -, wrt: rhs)
    @inlinable static func _vjpSubtract(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs - rhs, { Element.zero - $0 })
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable static func -=(lhs: inout Self, rhs: Element) {
        lhs = (lhs - rhs)
    }

    // VectorProtocol
    @differentiable(where Element: DifferentiableElement)
    @inlinable func subtracting(_ x: Element) -> Self {
        self - x
    }
}

//==============================================================================
/// mul
/// performs an elementwise multiply
/// - Parameters:
///  - lhs: left hand tensor
///  - rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@differentiable(where E: DifferentiableElement)
@inlinable public func mul<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E: Numeric
{
    /// REMOVE THIS
    let (lhs, rhs) = match(lhs, rhs)
    assert(lhs.shape == rhs.shape)

    var result = Tensor(like: lhs)
    Context.currentQueue.mul(lhs, rhs, &result)
    return result
}

@derivative(of: mul)
@inlinable func _vjpMultiply<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (lhs * rhs, { v in (v * rhs, v * lhs) })
}

public extension Tensor where Element: Numeric {
    @differentiable(where Element: DifferentiableElement)
    @inlinable static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }

    @inlinable static func *= (lhs: inout Self, rhs: Element) {
        lhs = mul(lhs, repeating(rhs, like: lhs))
    }

    @inlinable static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable static func * (lhs: Self, rhs: Element) -> Self {
        mul(lhs, repeating(rhs, like: lhs))
    }

    @derivative(of: *, wrt: lhs)
    @inlinable static func _vjpMultiply(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs * rhs, { $0 * rhs })
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: rhs where Element: DifferentiableElement)
    @inlinable static func * (lhs: Element, rhs: Self) -> Self {
        mul(repeating(lhs, like: rhs), rhs)
    }

    @derivative(of: *, wrt: rhs)
    @inlinable static func _vjpMultiply(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs * rhs, { $0 * rhs })
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func scaled(by scalar: Element) -> Self {
        self * scalar
    }

    // TODO: this syntax is incorrect and is only here to conform to
    // PointwiseMultiplicative and should be removed
    @differentiable(where Element: DifferentiableElement)
    @inlinable static func .* (lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }
}

//==============================================================================
/// div
/// performs an elementwise divide
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@differentiable(where E: DifferentiableElement)
@inlinable public func div<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E: AlgebraicField
{
    /// REMOVE THIS
    let (lhs, rhs) = match(lhs, rhs)
    assert(lhs.shape == rhs.shape)

    var result = Tensor(like: lhs)
    Context.currentQueue.div(lhs, rhs, &result)
    return result
}

@derivative(of: div)
@inlinable func _vjpDivide<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & AlgebraicField
{
    (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
}

public extension Tensor where Element: AlgebraicField {
    @differentiable(where Element: DifferentiableElement)
    @inlinable static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable static func /= (lhs: inout Self, rhs: Element) {
        lhs = div(lhs, repeating(rhs, like: lhs))
    }

    @inlinable static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable static func / (lhs: Self, rhs: Element) -> Self {
        div(lhs, Self(repeating: rhs, to: lhs.shape))
    }

    @derivative(of: /, wrt: lhs)
    @inlinable static func _vjpDivide(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs / rhs, { $0 / rhs })
    }

    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: rhs where Element: DifferentiableElement)
    @inlinable static func / (lhs: Element, rhs: Self) -> Self {
        div(Self(repeating: lhs, to: rhs.shape), rhs)
    }

    @derivative(of: /, wrt: rhs)
    @inlinable static func _vjpDivide(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where Element: DifferentiableElement {
        (lhs / rhs, { -lhs / rhs.squared() * $0 })
    }

    // PointwiseMultiplicative
    @differentiable(where Element: DifferentiableElement)
    @inlinable var reciprocal: Self {
        1 / self
    }
}
