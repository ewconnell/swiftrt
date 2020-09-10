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

extension Tensor where TensorElement.Value: AdditiveArithmetic {
    //--------------------------------------------------------------------------
    // tensor + tensor
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public static func +(lhs: Self, rhs: Self) -> Self {
        /// MAKE THIS GO AWAY!! assert(lhs.shape == rhs.shape) should be true
        /// Hack to work around AD zero materialization design problem
        if lhs.isZero {
            return rhs
        } else if rhs.isZero {
            return lhs
        } else {
            assert(lhs.shape == rhs.shape)
            var result = Tensor(like: lhs)
            Context.currentQueue.add(lhs, rhs, &result)
            return result
        }
    }

    @derivative(of: +)
    @usableFromInline static func _vjpAdd(_ lhs: Self, _ rhs: Self)
        -> (value: Self, pullback: (Self) -> (Self, Self)
    ) where Element: DifferentiableElement {
        (lhs + rhs, { ($0, $0) })
    }

    //--------------------------------------------------------------------------
    // tensor + Element
    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable public static func +(lhs: Self, rhs: Element) -> Self {
        var out = Tensor(like: lhs)
        Context.currentQueue.add(lhs, rhs, &out)
        return out
    }

    @derivative(of: +)
    @usableFromInline static func _vjpAdd(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableElement {
        (lhs + rhs, { ($0, $0.sum().element) })
    }

    @derivative(of: +, wrt: lhs)
    @usableFromInline static func _vjpAdd(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableElement {
        (lhs + rhs, { $0 })
    }
    
    // tensor += Element
    @differentiable(where Element: DifferentiableElement)
    @inlinable public static func +=(lhs: inout Self, rhs: Element) {
        lhs = lhs + rhs
    }
    
    //--------------------------------------------------------------------------
    // Element + tensor
    @differentiable(where Element: DifferentiableElement)
    @differentiable(wrt: rhs where Element: DifferentiableElement)
    @inlinable public static func +(lhs: Element, rhs: Self) -> Self {
        rhs + lhs
    }

    @derivative(of: +)
    @usableFromInline static func _vjpAdd(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableElement {
        (lhs + rhs, { ($0.sum().element, $0) })
    }

    @derivative(of: +, wrt: rhs)
    @usableFromInline static func _vjpAdd(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableElement {
        (lhs + rhs, { $0 })
    }
    
    //--------------------------------------------------------------------------
    // VectorProtocol
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public func adding(_ x: Element) -> Self {
        self + x
    }
}

//==============================================================================
/// subtract
extension Tensor where TensorElement.Value: AdditiveArithmetic {
    //--------------------------------------------------------------------------
    // tensor - tensor
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public static func -(lhs: Self, rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape)
        var result = Tensor(like: lhs)
        Context.currentQueue.subtract(lhs, rhs, &result)
        return result
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Self)
    -> (value: Self, pullback: (Self) -> (Self, Self)
    ) where Element: DifferentiableElement {
        (lhs - rhs, { ($0, 0 - $0) })
    }

    //--------------------------------------------------------------------------
    // tensor - Element
    @differentiable(where Element: DifferentiableElement & SignedNumeric)
    @differentiable(wrt: lhs where Element: DifferentiableElement)
    @inlinable public static func -(lhs: Self, rhs: Element) -> Self {
        var out = Tensor(like: lhs)
//        Context.currentQueue.subtract(lhs, rhs, &out)
        return out
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableElement & SignedNumeric {
        (lhs + rhs, { ($0, $0.sum().element) })
    }
    
    @derivative(of: -, wrt: lhs)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableElement {
        (lhs - rhs, { $0 })
    }

    @differentiable(where Element: DifferentiableElement & SignedNumeric)
    @inlinable public static func -=(lhs: inout Self, rhs: Element) {
        lhs = lhs - rhs
    }

    //--------------------------------------------------------------------------
    // Element - tensor
    @differentiable(where Element: DifferentiableElement & SignedNumeric)
    @differentiable(wrt: rhs where Element: DifferentiableElement & SignedNumeric)
    @inlinable public static func -(lhs: Element, rhs: Self) -> Self {
        var out = Tensor(like: rhs)
        //        Context.currentQueue.subtract(lhs, rhs, &out)
        return out
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableElement & SignedNumeric {
        (lhs + rhs, { ($0.sum().element, -$0) })
    }
    
    @derivative(of: -, wrt: rhs)
    @usableFromInline static func _vjpSubtract(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableElement & SignedNumeric {
        (lhs - rhs, { -$0 })
    }

    //--------------------------------------------------------------------------
    // VectorProtocol
    @differentiable(where Element: DifferentiableElement & SignedNumeric)
    @inlinable public func subtracting(_ x: Element) -> Self {
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
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func mul<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
{
    assert(lhs.shape == rhs.shape)
    var result = Tensor(like: lhs)
    Context.currentQueue.mul(lhs, rhs, &result)
    return result
}

@derivative(of: mul)
@usableFromInline func _vjpMultiply<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableElement
{
    (lhs * rhs, { v in (v * rhs, v * lhs) })
}

extension Tensor where TensorElement.Value: Numeric {
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }

    @inlinable public static func *= (lhs: inout Self, rhs: TensorElement.Value) {
        lhs = mul(lhs, repeating(rhs, like: lhs))
    }

    @inlinable public static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }

    //--------------------------------
    // tensor * Element
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @differentiable(wrt: lhs where TensorElement.Value: DifferentiableElement)
    @inlinable public static func * (lhs: Self, rhs: TensorElement.Value) -> Self {
        mul(lhs, repeating(rhs, like: lhs))
    }

    @derivative(of: *, wrt: lhs)
    @usableFromInline static func _vjpMultiply(_ lhs: Self, _ rhs: TensorElement.Value) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where TensorElement.Value: DifferentiableElement {
        (lhs * rhs, { $0 * rhs })
    }

    //--------------------------------
    // Element * tensor
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @differentiable(wrt: rhs where TensorElement.Value: DifferentiableElement)
    @inlinable public static func * (lhs: TensorElement.Value, rhs: Self) -> Self {
        mul(repeating(lhs, like: rhs), rhs)
    }

    @derivative(of: *, wrt: rhs)
    @usableFromInline static func _vjpMultiply(_ lhs: TensorElement.Value, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where TensorElement.Value: DifferentiableElement {
        (lhs * rhs, { lhs * $0 })
    }

    //--------------------------------
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public func scaled(by scalar: TensorElement.Value) -> Self {
        self * scalar
    }

    // TODO: this syntax is incorrect and is only here to conform to
    // PointwiseMultiplicative and should be removed
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public static func .* (lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }
}

//==============================================================================
/// div
/// performs an elementwise divide
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func div<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: AlgebraicField
{
    assert(lhs.shape == rhs.shape)
    var result = Tensor(like: lhs)
    Context.currentQueue.div(lhs, rhs, &result)
    return result
}

@derivative(of: div)
@usableFromInline func _vjpDivide<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) ->
    (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableElement & AlgebraicField
{
    (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
}

extension Tensor where TensorElement.Value: AlgebraicField {
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable public static func /= (lhs: inout Self, rhs: TensorElement.Value) {
        lhs = div(lhs, repeating(rhs, like: lhs))
    }

    @inlinable public static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    //--------------------------------
    // tensor / Element
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @differentiable(wrt: lhs where TensorElement.Value: DifferentiableElement)
    @inlinable public static func / (lhs: Self, rhs: TensorElement.Value) -> Self {
        div(lhs, repeating(rhs, like: lhs))
    }

    @derivative(of: /, wrt: lhs)
    @usableFromInline static func _vjpDivide(_ lhs: Self, _ rhs: TensorElement.Value) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where TensorElement.Value: DifferentiableElement {
        (lhs / rhs, { $0 / rhs })
    }

    //--------------------------------
    // Element / tensor
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @differentiable(wrt: rhs where TensorElement.Value: DifferentiableElement)
    @inlinable public static func / (lhs: TensorElement.Value, rhs: Self) -> Self {
        div(repeating(lhs, like: rhs), rhs)
    }

    @derivative(of: /, wrt: rhs)
    @usableFromInline static func _vjpDivide(_ lhs: TensorElement.Value, _ rhs: Self) -> (
        value: Self, pullback: (TangentVector) -> TangentVector
    ) where TensorElement.Value: DifferentiableElement {
        (lhs / rhs, { -lhs / rhs.squared() * $0 })
    }

    // PointwiseMultiplicative
    @differentiable(where TensorElement.Value: DifferentiableElement)
    @inlinable public var reciprocal: Self {
        1 / self
    }
}
