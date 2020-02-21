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
import Real

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
public func newAdd<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: AdditiveArithmetic
{
    Platform.service.newAdd(lhs, rhs)
}

extension PlatformService {
    //--------------------------------------------------------------------------
    @inlinable
    func newAdd<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    {
        let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var (result, resultBuffer) = createResultBuffer(like: lhs)
        currentQueue.newAdd(lhs: getBuffer(lhs),
                            rhs: getBuffer(rhs),
                            result: &resultBuffer)
        return result
    }
    
    //--------------------------------------------------------------------------

    @inlinable
    func add<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    {
        let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.add(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @derivative(of: add)
    @inlinable
    func _vjpAdd<T>(lhs: T, rhs: T) -> (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView
    {
        (lhs + rhs, { v in (v, v) })
    }
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable
    static func + (lhs: Self, rhs: Self) -> Self { add(lhs, rhs) }

    @inlinable
    static func += (lhs: inout Self, rhs: Element) { lhs = lhs + rhs }

    @inlinable
    static func +(lhs: Self, rhs: Element) -> Self {
        lhs + Self(repeating: rhs, like: lhs)
    }

    @inlinable
    static func +(lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) + rhs
    }
}

//--------------------------------------
// derivative functions
public extension TensorView where Self: DifferentiableTensorView {
    @derivative(of: +)
    @inlinable
    static func _vjpAdd(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        Platform.service._vjpAdd(lhs: lhs, rhs: rhs)
    }
    
    @derivative(of: +)
    @inlinable
    static func _vjpAdd(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        (lhs + rhs, { v in (v, v.sum().element) })
    }
    
    @derivative(of: +)
    @inlinable
    static func _vjpAdd(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        (lhs + rhs, { v in (v.sum().element, v) })
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

extension PlatformService {
    @inlinable
    func subtract<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AdditiveArithmetic
    {
        let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.subtract(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    
    @derivative(of: subtract)
    @inlinable
    func _vjpSubtract<T>(lhs: T, rhs: T) -> (value: T, pullback: (T) ->(T, T))
        where T: DifferentiableTensorView
    {
        (lhs - rhs, { v in (v, v) })
    }
}

public extension TensorView where Element: AdditiveArithmetic {
    @inlinable
    static func - (lhs: Self, rhs: Self) -> Self { subtract(lhs, rhs) }

    @inlinable
    static func -= (lhs: inout Self, rhs: Element) { lhs = lhs - rhs }
    
    @inlinable
    static func - (lhs: Self, rhs: Element) -> Self {
        lhs - Self(repeating: rhs, like: lhs)
    }

    @inlinable
    static func - (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) - rhs
    }
}

//--------------------------------------
// derivative functions
public extension TensorView
    where Self: DifferentiableTensorView, Element: SignedNumeric
{
    @derivative(of: -)
    @inlinable
    static func vjpSubtract(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        return (lhs - rhs, { v in (v, -v) })
    }
    
    @derivative(of: -)
    @inlinable
    static func vjpSubtract(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        return (lhs - rhs, { v in (v, -v.sum().element) })
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

extension PlatformService {
    @inlinable
    func mul<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: Numeric
    {
        let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.mul(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @derivative(of: mul)
    @inlinable
    internal func _vjpMultiply<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where T: DifferentiableTensorView
    {
        (lhs * rhs, { v in (v * rhs, v * lhs) })
    }
}

public extension TensorView where Element: Numeric {
    @inlinable
    static func * (lhs: Self, rhs: Self) -> Self { mul(lhs, rhs) }
    
    @inlinable
    static func *= (lhs: inout Self, rhs: Element) { lhs = lhs * rhs }

    @inlinable
    static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
    
    @inlinable
    static func * (lhs: Self, rhs: Element) -> Self {
        lhs * Self(repeating: rhs, like: lhs)
    }

    @inlinable
    static func * (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) * rhs
    }
}

//--------------------------------------
// derivative functions
public extension TensorView where Self: DifferentiableTensorView {
    @derivative(of: *)
    @inlinable
    static func _vjpMultiply(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        Platform.service._vjpMultiply(lhs, rhs)
    }
    
    @derivative(of: *)
    @inlinable
    static func _vjpMultiply(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        (lhs * rhs, { v in (v * rhs, (v * lhs).sum().element) })
    }
    
    @derivative(of: *)
    @inlinable
    static func _vjpMultiply(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        (lhs * rhs, { v in ((v * rhs).sum().element, v * lhs) })
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

extension PlatformService {
    @inlinable
    func div<T>(_ lhs: T, _ rhs: T) -> T
        where T: TensorView, T.Element: AlgebraicField
    {
        let (lhs, rhs) = implicitlyMatchExtents(lhs, rhs)
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.div(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @derivative(of: div)
    @inlinable
    internal func _vjpDivide<T>(_ lhs: T, _ rhs: T) ->
        (value: T, pullback: (T) -> (T, T)) where
        T: DifferentiableTensorView, T.Element: AlgebraicField & SignedNumeric
    {
        (lhs / rhs, { v in (v / rhs, -lhs / rhs.squared() * v) })
    }
}

public extension TensorView where Element: AlgebraicField {
    @inlinable
    static func / (lhs: Self, rhs: Self) -> Self { div(lhs, rhs) }

    @inlinable
    static func /= (lhs: inout Self, rhs: Element) { lhs = lhs / rhs }

    @inlinable
    static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }

    @inlinable
    static func / (lhs: Self, rhs: Element) -> Self {
        lhs / Self(repeating: rhs, like: lhs)
    }

    @inlinable
    static func / (lhs: Element, rhs: Self) -> Self {
        Self(repeating: lhs, like: rhs) / rhs
    }
}

//--------------------------------------
// derivative functions
public extension TensorView where
    Self: DifferentiableTensorView,
    Element: AlgebraicField & SignedNumeric
{
    @derivative(of: /)
    @inlinable
    static func _vjpDivide(lhs: Self, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Self, Self))
    {
        Platform.service._vjpDivide(lhs, rhs)
    }
    
    @derivative(of: /)
    @inlinable
    static func _vjpDivide(lhs: Self, rhs: Element) ->
        (value: Self, pullback: (Self) -> (Self, Element))
    {
        (lhs / rhs, { v in
            (v / rhs, (-lhs / rhs.squared() * v).sum().element)
        })
    }
    
    @derivative(of: /)
    @inlinable
    static func _vjpDivide(lhs: Element, rhs: Self) ->
        (value: Self, pullback: (Self) -> (Element, Self))
    {
        (lhs / rhs, { v in
            ((v / rhs).sum().element, -lhs / rhs.squared() * v)
        })
    }
}
