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
public func add<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AdditiveArithmetic
{
    Platform.service.add(lhs, rhs)
}

@inlinable
@derivative(of: add)
func _vjpAdd<T>(lhs: T, rhs: T) ->
    (value: T, pullback: (T) ->(T, T)) where T: DifferentiableTensorView
{
    Platform.service._vjpAdd(lhs, rhs)
}

public extension PlatformService {
    @inlinable
    func add<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    {
        let (left, right) = implicitlyMatchExtents(lhs, rhs)
        assert(left.extents == right.extents, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResult(like: left)
        currentQueue.add(read(left), read(right), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: add)
    func _vjpAdd<T>(_ lhs: T, _ rhs: T) -> (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView
    {
        (lhs + rhs, { v in (v, v) })
    }
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func + (lhs: Self, rhs: Self) -> Self { add(lhs, rhs) }

    @inlinable
    static func += (lhs: inout Self, rhs: Element) {
        lhs = add(lhs, Self(repeating: rhs, like: lhs))
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func +(lhs: Self, rhs: Element) -> Self {
        add(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func +(lhs: Element, rhs: Self) -> Self {
        add(Self(repeating: lhs, to: rhs.extents), rhs)
    }
}

//==============================================================================
/// subtract
/// peforms an elementwise subtract
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable
public func subtract<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AdditiveArithmetic
{
    Platform.service.subtract(lhs, rhs)
}

@derivative(of: subtract)
@inlinable
public func _vjpSubtract<T>(lhs: T, rhs: T) ->
    (value: T, pullback: (T) ->(T, T)) where T: DifferentiableTensorView
{
    Platform.service._vjpSubtract(lhs, rhs)
}

public extension PlatformService {
    @inlinable
    func subtract<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    {
        let (left, right) = implicitlyMatchExtents(lhs, rhs)
        assert(left.extents == right.extents, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResult(like: left)
        currentQueue.subtract(read(left), read(right), &resultBuffer)
        return result
    }
        
    @inlinable
    @derivative(of: subtract)
    func _vjpSubtract<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView
    {
        (lhs - rhs, { v in (v, T.zero - v) })
    }
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func - (lhs: Self, rhs: Self) -> Self { subtract(lhs, rhs) }

    @inlinable
    static func -= (lhs: inout Self, rhs: Element) {
        lhs = subtract(lhs, Self(repeating: rhs, like: lhs))
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func - (lhs: Self, rhs: Element) -> Self {
        subtract(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func - (lhs: Element, rhs: Self) -> Self {
        subtract(Self(repeating: lhs, to: rhs.extents), rhs)
    }
}

//==============================================================================
/// mul
/// performs an elementwise multiply
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable
public func mul<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.mul(lhs, rhs)
}

@inlinable
@derivative(of: mul)
func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
    (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensorView
{
    Platform.service._vjpMultiply(lhs, rhs)
}

public extension PlatformService {
    @inlinable
    func mul<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: Numeric
    {
        let (left, right) = implicitlyMatchExtents(lhs, rhs)
        assert(left.extents == right.extents, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResult(like: left)
        currentQueue.mul(read(left), read(right), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: mul)
    func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensorView
    {
        (lhs * rhs, { v in (v * rhs, v * lhs) })
    }
}

public extension TensorView where Element: Numeric {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }
    
    @inlinable
    static func *= (lhs: inout Self, rhs: Element) {
        lhs = mul(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func * (lhs: Self, rhs: Element) -> Self {
        mul(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func * (lhs: Element, rhs: Self) -> Self {
        mul(Self(repeating: lhs, to: rhs.extents), rhs)
    }
}

//==============================================================================
/// div
/// performs an elementwise divide
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable
public func div<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AlgebraicField
{
    Platform.service.div(lhs, rhs)
}

@inlinable
@derivative(of: div)
func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
    (value: T, pullback: (T) -> (T, T)) where
    T: DifferentiableTensorView, T.Element: AlgebraicField
{
    Platform.service._vjpDivide(lhs, rhs)
}

public extension PlatformService {
    @inlinable
    func div<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AlgebraicField
    {
        let (left, right) = implicitlyMatchExtents(lhs, rhs)
        assert(left.extents == right.extents, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResult(like: left)
        currentQueue.div(read(left), read(right), &resultBuffer)
        return result
    }
    
    @inlinable
    @derivative(of: div)
    func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where
        T: DifferentiableTensorView, T.Element: AlgebraicField
    {
        (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
    }
}

public extension TensorView where Element: AlgebraicField {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable
    static func /= (lhs: inout Self, rhs: Element) {
        lhs = div(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func / (lhs: Self, rhs: Element) -> Self {
        div(lhs, Self(repeating: rhs, to: lhs.extents))
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    static func / (lhs: Element, rhs: Self) -> Self {
        div(Self(repeating: lhs, to: rhs.extents), rhs)
    }
}
