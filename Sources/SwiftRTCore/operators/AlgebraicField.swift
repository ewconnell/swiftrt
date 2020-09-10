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

extension Tensor where Element: AdditiveArithmetic {
    //==============================================================================
    // add
    // tensor + tensor
    @differentiable(where Element: DifferentiableNumeric)
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
    ) where Element: DifferentiableNumeric {
        (lhs + rhs, { ($0, $0) })
    }

    //--------------------------------------------------------------------------
    // tensor + Element
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: lhs where Element: DifferentiableNumeric)
    @inlinable public static func +(lhs: Self, rhs: Element) -> Self {
        var out = Tensor(like: lhs)
        Context.currentQueue.add(lhs, rhs, &out)
        return out
    }

    @derivative(of: +)
    @usableFromInline static func _vjpAdd(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableNumeric {
        (lhs + rhs, { ($0, $0.sum().element) })
    }

    @derivative(of: +, wrt: lhs)
    @usableFromInline static func _vjpAdd(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs + rhs, { $0 })
    }
    
    // tensor += Element
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public static func +=(lhs: inout Self, rhs: Element) {
        lhs = lhs + rhs
    }
    
    //--------------------------------------------------------------------------
    // Element + tensor
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: rhs where Element: DifferentiableNumeric)
    @inlinable public static func +(lhs: Element, rhs: Self) -> Self {
        rhs + lhs
    }

    @derivative(of: +)
    @usableFromInline static func _vjpAdd(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableNumeric {
        (lhs + rhs, { ($0.sum().element, $0) })
    }

    @derivative(of: +, wrt: rhs)
    @usableFromInline static func _vjpAdd(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs + rhs, { $0 })
    }
    
    //--------------------------------------------------------------------------
    // VectorProtocol
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public func adding(_ x: Element) -> Self {
        self + x
    }
}

//==============================================================================
/// subtract
extension Tensor where Element: AdditiveArithmetic {
    //--------------------------------------------------------------------------
    // tensor - tensor
    @differentiable(where Element: DifferentiableNumeric & SignedNumeric)
    @inlinable public static func -(lhs: Self, rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape)
        var result = Tensor(like: lhs)
        Context.currentQueue.subtract(lhs, rhs, &result)
        return result
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Self)
    -> (value: Self, pullback: (Self) -> (Self, Self)
    ) where Element: DifferentiableNumeric & SignedNumeric {
        (lhs - rhs, { ($0, -$0) })
    }

    //--------------------------------------------------------------------------
    // tensor - Element
    @differentiable(where Element: DifferentiableNumeric & SignedNumeric)
    @differentiable(wrt: lhs where Element: DifferentiableNumeric)
    @inlinable public static func -(lhs: Self, rhs: Element) -> Self {
        var out = Tensor(like: lhs)
        Context.currentQueue.subtract(lhs, rhs, &out)
        return out
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableNumeric & SignedNumeric {
        (lhs + rhs, { ($0, $0.sum().element) })
    }
    
    @derivative(of: -, wrt: lhs)
    @usableFromInline static func _vjpSubtract(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs - rhs, { $0 })
    }

    @differentiable(where Element: DifferentiableNumeric & SignedNumeric)
    @inlinable public static func -=(lhs: inout Self, rhs: Element) {
        lhs = lhs - rhs
    }

    //--------------------------------------------------------------------------
    // Element - tensor
    @differentiable(where Element: DifferentiableNumeric & SignedNumeric)
    @differentiable(wrt: rhs where Element: DifferentiableNumeric & SignedNumeric)
    @inlinable public static func -(lhs: Element, rhs: Self) -> Self {
        var out = Tensor(like: rhs)
        Context.currentQueue.subtract(lhs, rhs, &out)
        return out
    }

    @derivative(of: -)
    @usableFromInline static func _vjpSubtract(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableNumeric & SignedNumeric {
        (lhs + rhs, { ($0.sum().element, -$0) })
    }
    
    @derivative(of: -, wrt: rhs)
    @usableFromInline static func _vjpSubtract(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric & SignedNumeric {
        (lhs - rhs, { -$0 })
    }

    //--------------------------------------------------------------------------
    // VectorProtocol
    @differentiable(where Element: DifferentiableNumeric & SignedNumeric)
    @inlinable public func subtracting(_ x: Element) -> Self {
        self - x
    }
}

//==============================================================================
/// mul
extension Tensor where Element: Numeric {
    //--------------------------------------------------------------------------
    // tensor * tensor
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public static func * (lhs: Self, rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape)
        var out = Tensor(like: lhs)
        Context.currentQueue.mul(lhs, rhs, &out)
        return out
    }

    @derivative(of: *)
    @usableFromInline static func _vjpMultiply(_ lhs: Self, _ rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self)
    ) where Element: DifferentiableNumeric {
        (lhs * rhs, { v in (v * rhs, v * lhs) })
    }

    @inlinable public static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }
    
    //--------------------------------------------------------------------------
    // tensor * Element
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: lhs where Element: DifferentiableNumeric)
    @inlinable public static func * (lhs: Self, rhs: Element) -> Self {
        var out = Tensor(like: lhs)
        Context.currentQueue.mul(lhs, rhs, &out)
        return out
    }

    @derivative(of: *)
    @usableFromInline static func _vjpMultiply(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableNumeric {
        (lhs * rhs, { ($0 * rhs, ($0 * lhs).sum().element) })
    }

    @derivative(of: *, wrt: lhs)
    @usableFromInline static func _vjpMultiply(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs * rhs, { $0 * rhs })
    }

    @inlinable public static func *= (lhs: inout Self, rhs: Element) {
        lhs = lhs * rhs
    }
    
    //--------------------------------------------------------------------------
    // Element * tensor
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: rhs where Element: DifferentiableNumeric)
    @inlinable public static func * (lhs: Element, rhs: Self) -> Self {
        var out = Tensor(like: rhs)
        Context.currentQueue.mul(rhs, lhs, &out)
        return out
    }

    @derivative(of: *)
    @usableFromInline static func _vjpMultiply(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableNumeric {
        (lhs * rhs, { (($0 * rhs).sum().element, $0 * lhs) })
    }
    
    @derivative(of: *, wrt: rhs)
    @usableFromInline static func _vjpMultiply(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs * rhs, { lhs * $0 })
    }

    //--------------------------------
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public func scaled(by scalar: Element) -> Self {
        self * scalar
    }

    // TODO: this syntax is incorrect and is only here to conform to
    // PointwiseMultiplicative and should be removed
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public static func .* (lhs: Self, rhs: Self) -> Self {
        lhs * rhs
    }
}

//==============================================================================
/// div
extension Tensor where Element: AlgebraicField {
    //--------------------------------------------------------------------------
    // tensor / tensor
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public static func / (lhs: Self, rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape)
        var result = Tensor(like: lhs)
        Context.currentQueue.div(lhs, rhs, &result)
        return result
    }

    @derivative(of: /)
    @usableFromInline static func _vjpDivide(_ lhs: Self, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Self, Self)
    ) where Element: DifferentiableNumeric & AlgebraicField {
        (lhs / rhs, { ($0 / rhs, -lhs / rhs.squared() * $0) })
    }

    @inlinable public static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
    }

    //--------------------------------
    // tensor / Element
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: lhs where Element: DifferentiableNumeric)
    @inlinable public static func / (lhs: Self, rhs: Element) -> Self {
        var result = Tensor(like: lhs)
        Context.currentQueue.div(lhs, rhs, &result)
        return result
    }

    @derivative(of: /)
    @usableFromInline static func _vjpDivide(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> (Self, Element)
    ) where Element: DifferentiableNumeric {
        (lhs / rhs, { ($0 / rhs, ($0 * -lhs / rhs.squared()).sum().element) })
    }

    @derivative(of: /, wrt: lhs)
    @usableFromInline static func _vjpDivide(_ lhs: Self, _ rhs: Element) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs / rhs, { $0 / rhs })
    }

    @inlinable public static func /= (lhs: inout Self, rhs: Element) {
        lhs = lhs / rhs
    }

    //--------------------------------
    // Element / tensor
    @differentiable(where Element: DifferentiableNumeric)
    @differentiable(wrt: rhs where Element: DifferentiableNumeric)
    @inlinable public static func / (lhs: Element, rhs: Self) -> Self {
        var result = Tensor(like: rhs)
        Context.currentQueue.div(lhs, rhs, &result)
        return result
    }

    @derivative(of: /)
    @usableFromInline static func _vjpDivide(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> (Element, Self)
    ) where Element: DifferentiableNumeric {
        (lhs / rhs, { (($0 / rhs).sum().element, $0 * -lhs / rhs.squared()) })
    }
    
    @derivative(of: /, wrt: rhs)
    @usableFromInline static func _vjpDivide(_ lhs: Element, _ rhs: Self) -> (
        value: Self, pullback: (Self) -> Self
    ) where Element: DifferentiableNumeric {
        (lhs / rhs, { -lhs / rhs.squared() * $0 })
    }

    // PointwiseMultiplicative
    @differentiable(where Element: DifferentiableNumeric)
    @inlinable public var reciprocal: Self {
        1 / self
    }
}
