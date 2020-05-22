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
/// and
/// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
@inlinable public func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.and(lhs, rhs, &result)
    return result
}

@inlinable public func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Bool)
    -> Tensor<S,Bool> where S: TensorShape
{
    and(lhs, repeating(rhs, like: lhs))
}

@inlinable public func and<S>(_ lhs: Bool, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    and(repeating(lhs, like: rhs), rhs)
}

extension Tensor where Element == Bool {
    @inlinable
    public static func .&&(_ lhs: Self, _ rhs: Self) -> Self { and(lhs, rhs) }

    @inlinable
    public static func .&&(_ lhs: Self, _ rhs: Element) -> Self { and(lhs, rhs) }

    @inlinable
    public static func .&&(_ lhs: Element, _ rhs: Self) -> Self { and(lhs, rhs) }
}

//==============================================================================
/// or
/// Computes `lhs .|| rhs` element-wise and returns a tensor of Bool values
@inlinable public func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.or(lhs, rhs, &result)
    return result
}

@inlinable public func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Bool)
    -> Tensor<S,Bool> where S: TensorShape
{
    or(lhs, repeating(rhs, like: lhs))
}

@inlinable public func or<S>(_ lhs: Bool, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    or(repeating(lhs, like: rhs), rhs)
}

public extension Tensor where Element == Bool {
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Self) -> Self { or(lhs, rhs) }

    @inlinable
    static func .||(_ lhs: Self, _ rhs: Element) -> Self { or(lhs, rhs) }

    @inlinable
    static func .||(_ lhs: Element, _ rhs: Self) -> Self { or(lhs, rhs) }
}

//==============================================================================
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.max(lhs, rhs, &result)
    return result
}

@derivative(of: max)
@inlinable func _vjpMax<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Comparable
{
    (value: max(lhs, rhs), {
        var resultTrue = Tensor(like: lhs)
        var resultFalse = Tensor(like: lhs)
        Context.currentQueue.vjpMax(lhs, rhs, $0, &resultTrue, &resultFalse)
        return (resultTrue, resultFalse)
    })
}

//--------------------------------
// tensor Element
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    max(lhs, repeating(rhs, like: lhs))
}

@derivative(of: max, wrt: lhs)
@inlinable public func _vjpMax<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
-> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E: Comparable & Numeric & DifferentiableElement
{
    let element = repeating(rhs, like: lhs)
    return (value: max(lhs, element), {
        var result: Tensor<S,E> = Tensor(like: $0)
        Context.currentQueue.vjpMax(lhs, element, $0, &result)
        return result
    })
}

//--------------------------------
// Element tensor
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func max<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    max(repeating(lhs, like: rhs), rhs)
}

@derivative(of: max, wrt: rhs)
@inlinable public func _vjpMax<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
-> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E: Comparable & Numeric & DifferentiableElement
{
    let element = repeating(lhs, like: rhs)
    return (value: max(element, rhs), {
        var result: Tensor<S,E> = Tensor(like: $0)
        // tensor should be first parameter
        Context.currentQueue.vjpMax(rhs, element, $0, &result)
        return result
    })
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor where Element: Comparable {
    @differentiable(where Element: DifferentiableElement)
    @inlinable func max(_ lhs: Self, _ rhs: Self) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func max(_ lhs: Self, _ rhs: Element) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func max(_ lhs: Element, _ rhs: Self) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }
}

//==============================================================================
/// min
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.min(lhs, rhs, &result)
    return result
}

@derivative(of: min)
@inlinable func _vjpMin<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Comparable
{
    (value: min(lhs, rhs), {
        var resultTrue = Tensor(like: lhs)
        var resultFalse = Tensor(like: lhs)
        Context.currentQueue.vjpMin(lhs, rhs, $0, &resultTrue, &resultFalse)
        return (resultTrue, resultFalse)
    })
}

//--------------------------------
// tensor Element
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    min(lhs, repeating(rhs, like: lhs))
}

@derivative(of: min, wrt: lhs)
@inlinable public func _vjpMin<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
-> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E: Comparable & Numeric & DifferentiableElement
{
    let element = repeating(rhs, like: lhs)
    return (value: min(lhs, element), {
        var result: Tensor<S,E> = Tensor(like: $0)
        Context.currentQueue.vjpMin(lhs, element, $0, &result)
        return result
    })
}

//--------------------------------
// Element tensor
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func min<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    min(repeating(lhs, like: rhs), rhs)
}

@derivative(of: min, wrt: rhs)
@inlinable public func _vjpMin<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
-> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E: Comparable & Numeric & DifferentiableElement
{
    let element = repeating(lhs, like: rhs)
    return (value: min(element, rhs), {
        var result: Tensor<S,E> = Tensor(like: $0)
        // tensor should be first parameter
        Context.currentQueue.vjpMin(rhs, element, $0, &result)
        return result
    })
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor where Element: Comparable {
    @differentiable(where Element: DifferentiableElement)
    @inlinable func min(_ lhs: Self, _ rhs: Self) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func min(_ lhs: Self, _ rhs: Element) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }

    @differentiable(where Element: DifferentiableElement)
    @inlinable func min(_ lhs: Element, _ rhs: Self) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }
}

//==============================================================================
/// equal
/// Performs element-wise equality comparison and returns a
/// tensor of Bool values
@inlinable
public func equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,Bool>
where S: TensorShape, E: Equatable
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor<S, Bool>(lhs.shape)
    Context.currentQueue.equal(lhs, rhs, &result)
    return result
}

extension Tensor: Equatable where Element: Equatable {
    @inlinable
    public static func .== (_ lhs: Self, _ rhs: Self) -> Tensor<Shape, Bool> {
        equal(lhs, rhs)
    }

    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable public static func == (lhs: Self, rhs: Self) -> Bool {
        // the bounds must match or they are not equal
        guard lhs.shape == rhs.shape else { return false }

        // if lhs is an alias for rhs, then they match
        if lhs.storage === rhs.storage && lhs.baseOffset == rhs.baseOffset {
            return true
        }

        // compare elements
        return (lhs .== rhs).all().element
    }
}

//==============================================================================
/// elementsAlmostEqual
/// Performs element-wise equality comparison within the tolerance range
/// and returns a tensor of Bool values
@inlinable public func elementsAlmostEqual<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>,
    tolerance: E.Value
) -> Tensor<S,Bool>
where S: TensorShape, E.Value: SignedNumeric & Comparable {
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.elementsAlmostEqual(lhs, rhs, tolerance, &result)
    return result
}

public extension Tensor where TensorElement.Value: SignedNumeric & Comparable {
    @inlinable func elementsAlmostEqual(_ rhs: Self, tolerance: Element)
        -> Tensor<Shape,Bool>
    {
        SwiftRTCore.elementsAlmostEqual(self, rhs, tolerance: tolerance)
    }
}

//==============================================================================
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
@inlinable public func notEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,Bool> where S: TensorShape, E: Equatable
{
    assert(lhs.shape == rhs.shape, _messageTensorExtentsMismatch)
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.notEqual(lhs, rhs, &result)
    return result
}

public extension Tensor where Element: Equatable {
    @inlinable static func .!=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape, Bool> {
        notEqual(lhs, rhs)
    }
}

//==============================================================================
/// greater
/// Computes `lhs .> rhs` element-wise and returns a tensor of Bool values
@inlinable
public func greater<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,Bool> where S: TensorShape, E: Comparable
{
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.greater(lhs, rhs, &result)
    return result
}

public extension Tensor where Element: Comparable {
    @inlinable static func .>(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        greater(lhs, rhs)
    }
}

//==============================================================================
/// greaterOrEqual
/// Computes `lhs .>= rhs` element-wise and returns a tensor of Bool values
@inlinable
public func greaterOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S, Bool> where S: TensorShape, E: Comparable
{
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.greaterOrEqual(lhs, rhs, &result)
    return result
}

public extension Tensor where Element: Comparable {
    @inlinable static func .>=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        greaterOrEqual(lhs, rhs)
    }
}

//==============================================================================
/// less
/// Computes `lhs .< rhs` element-wise and returns a tensor of Bool values
@inlinable public func less<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,Bool> where S: TensorShape, E: Comparable
{
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.less(lhs, rhs, &result)
    return result
}

public extension Tensor where Element: Comparable {
    @inlinable static func .<(_ lhs: Self, _ rhs: Self) -> Tensor<Shape, Bool> {
        less(lhs, rhs)
    }
}

//==============================================================================
/// lessOrEqual
/// Computes `lhs .<= rhs` element-wise and returns a tensor of Bool values
@inlinable public func lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,Bool> where S: TensorShape, E: Comparable
{
    var result = Tensor<S,Bool>(lhs.shape)
    Context.currentQueue.lessOrEqual(lhs, rhs, &result)
    return result
}

@inlinable public func lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
    -> Tensor<S,Bool> where S: TensorShape, E: Comparable
{
    lessOrEqual(lhs, repeating(rhs, like: lhs))
}

@inlinable public func lessOrEqual<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
    -> Tensor<S,Bool> where S: TensorShape, E: Comparable
{
    lessOrEqual(repeating(lhs, like: rhs), rhs)
}

public extension Tensor where Element: Comparable {
    @inlinable static func .<=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Self, _ rhs: Element) -> Tensor<Shape,Bool> {
        lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Element, _ rhs: Self) -> Tensor<Shape,Bool> {
        lessOrEqual(lhs, rhs)
    }
}
