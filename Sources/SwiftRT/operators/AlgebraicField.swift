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
@inlinable
public func add<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    var result = Tensor(like: lhs)
    Context.currentQueue.add(lhs, rhs, &result)
    return result
}

//@differentiable(where T: DifferentiableTensor)
@inlinable public func add<S,E>(_ lhs: Tensor<S,E>, _ rhs: E) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    add(lhs, repeating(rhs, like: lhs))
}

//@differentiable(where T: DifferentiableTensor)
@inlinable public func add<S, E>(_ lhs: E, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    add(repeating(lhs, like: rhs), rhs)
}

//@derivative(of: add)
@inlinable public func _vjpAdd<S, E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) ->(Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (lhs + rhs, { v in (v, v) })
}

public extension Tensor where Element: AdditiveArithmetic {
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +(lhs: Self, rhs: Self) -> Self {
        add(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +(lhs: Self, rhs: Element) -> Self {
        add(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func +(lhs: Element, rhs: Self) -> Self {
        add(lhs, rhs)
    }

    // VectorProtocol
    
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable func adding(_ x: Element) -> Self {
        self + x
    }
}

//==============================================================================
/// subtract
/// performs an elementwise subtract
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable
//@differentiable(where T: DifferentiableTensor)
public func subtract<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    var result = Tensor(like: lhs)
    Context.currentQueue.subtract(lhs, rhs, &result)
    return result
}

@inlinable
public func subtract<S,E>(_ lhs: Tensor<S,E>, _ rhs: E) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    subtract(lhs, repeating(rhs, like: lhs))
}

//@differentiable(where T: DifferentiableTensor)
@inlinable public func subtract<S, E>(_ lhs: E, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: AdditiveArithmetic
{
    subtract(repeating(lhs, like: rhs), rhs)
}

//@derivative(of: add)
@inlinable public func _vjpSubtract<S, E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) ->(Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (lhs - rhs, { v in (v, E.zero - v) })
}

public extension Tensor where Element: AdditiveArithmetic {
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func -(lhs: Self, rhs: Self) -> Self {
        subtract(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func -(lhs: Self, rhs: Element) -> Self {
        subtract(lhs, rhs)
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func -(lhs: Element, rhs: Self) -> Self {
        subtract(lhs, rhs)
    }

    // VectorProtocol
    
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable func subtracting(_ x: Element) -> Self {
        self - x
    }
}

//==============================================================================
/// mul
/// performs an elementwise multiply
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable
public func mul<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: Numeric
{
    var result = Tensor(like: lhs)
    Context.currentQueue.mul(lhs, rhs, &result)
    return result
}

@inlinable
//@derivative(of: mul)
func _vjpMultiply<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement
{
    (lhs * rhs, { v in (v * rhs, v * lhs) })
}

public extension Tensor where Element: Numeric
{
//    @differentiable(where Element: DifferentiableElement)
    @inlinable static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }

    @inlinable static func *= (lhs: inout Self, rhs: Element) {
        lhs = mul(lhs, repeating(rhs, like: lhs))
    }

    @inlinable static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }

//    @differentiable(where Element: DifferentiableElement)
    @inlinable static func * (lhs: Self, rhs: Element) -> Self {
        mul(lhs, repeating(rhs, like: lhs))
    }

//    @differentiable(where Element: DifferentiableElement)
    @inlinable static func * (lhs: Element, rhs: Self) -> Self {
        mul(repeating(lhs, like: rhs), rhs)
    }

//    @differentiable(where Element: DifferentiableElement)
    @inlinable func scaled(by scalar: Element) -> Self {
        self * scalar
    }

    // TODO: this syntax is incorrect and is only here to conform to
    // PointwiseMultiplicative and should be removed
//    @differentiable(where Element: DifferentiableElement)
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
@inlinable
public func div<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
    where S: TensorShape, E: AlgebraicField
{
    var result = Tensor(like: lhs)
    Context.currentQueue.div(lhs, rhs, &result)
    return result
}

@inlinable
//@derivative(of: div)
func _vjpDivide<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & AlgebraicField
{
    (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
}

public extension Tensor where Element: AlgebraicField {
    
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable static func /= (lhs: inout Self, rhs: Element) {
        lhs = div(lhs, repeating(rhs, like: lhs))
    }

    @inlinable static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func / (lhs: Self, rhs: Element) -> Self {
        div(lhs, repeating(rhs, like: lhs))
    }

//    @differentiable(where Self: DifferentiableTensor)
    @inlinable static func / (lhs: Element, rhs: Self) -> Self {
        div(repeating(lhs, like: rhs), rhs)
    }

    // PointwiseMultiplicative
//    @differentiable(where Self: DifferentiableTensor)
    @inlinable var reciprocal: Self {
        1 / self
    }
}
