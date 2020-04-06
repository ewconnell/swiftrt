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
// utilities
@inlinable
func _vjpMinMax<S,E>(
    _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
    _ op: @escaping (E, E) -> Bool) -> (Tensor<S,E>, Tensor<S,E>)
    where S: TensorShape, E: Comparable & Numeric
{
    var resultTrue = empty(like: x)
    var resultFalse = empty(like: x)
    Context.currentQueue.vjpMinMax(x, y, scale, op, &resultTrue, &resultFalse)
    return (resultTrue, resultFalse)
}

//==============================================================================
/// and
/// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
@inlinable public func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    var result = empty(like: lhs)
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

infix operator .&& : LogicalConjunctionPrecedence

public extension Tensor where Element == Bool {
    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Self) -> Self { and(lhs, rhs) }

    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Element) -> Self { and(lhs, rhs) }

    @inlinable
    static func .&&(_ lhs: Element, _ rhs: Self) -> Self { and(lhs, rhs) }
}

//==============================================================================
/// or
/// Computes `lhs .|| rhs` element-wise and returns a tensor of Bool values
@inlinable public func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>)
    -> Tensor<S,Bool> where S: TensorShape
{
    var result = empty(like: lhs)
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

infix operator .|| : LogicalConjunctionPrecedence

public extension Tensor where Element == Bool {
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Self) -> Self { and(lhs, rhs) }

    @inlinable
    static func .||(_ lhs: Self, _ rhs: Element) -> Self { and(lhs, rhs) }

    @inlinable
    static func .||(_ lhs: Element, _ rhs: Self) -> Self { and(lhs, rhs) }
}

//==============================================================================
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
//@differentiable(where T: DifferentiableTensorView)
@inlinable public func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    var result = empty(like: lhs)
    Context.currentQueue.max(lhs, rhs, &result)
    return result
}

//@derivative(of: max)
@inlinable func _vjpMax<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Comparable
{
    (value: max(lhs, rhs), { _vjpMinMax(lhs, rhs, $0, >=) })
}

//@differentiable(where T: DifferentiableTensorView)
@inlinable public func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    max(lhs, repeating(rhs, like: lhs))
}

//@differentiable(where T: DifferentiableTensorView)
@inlinable public func max<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    max(repeating(lhs, like: rhs), rhs)
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor {
//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { SwiftRT.max(lhs, rhs) }

//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: E) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { max(lhs, repeating(rhs, like: lhs)) }

//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func max<S,E>(_ lhs: E, _ rhs: Tensor<S,E>) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { max(repeating(lhs, like: rhs), rhs) }
}

//==============================================================================
/// min
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
//@differentiable(where T: DifferentiableTensorView)
@inlinable public func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    var result = empty(like: lhs)
    Context.currentQueue.min(lhs, rhs, &result)
    return result
}

//@derivative(of: max)
@inlinable func _vjpMin<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>)
    -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E: DifferentiableElement & Comparable
{
    (value: min(lhs, rhs), { _vjpMinMax(lhs, rhs, $0, <=) })
}

//@differentiable(where T: DifferentiableTensorView)
@inlinable public func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: E)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    min(lhs, repeating(rhs, like: lhs))
}

//@differentiable(where T: DifferentiableTensorView)
@inlinable public func min<S,E>(_ lhs: E, _ rhs: Tensor<S,E>)
    -> Tensor<S,E> where S: TensorShape, E: Comparable
{
    min(repeating(lhs, like: rhs), rhs)
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor {
//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { SwiftRT.min(lhs, rhs) }

//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: E) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { min(lhs, repeating(rhs, like: lhs)) }

//    @differentiable(where T: DifferentiableTensorView)
    @inlinable func min<S,E>(_ lhs: E, _ rhs: Tensor<S,E>) -> Tensor<S,E>
        where S: TensorShape, E: Comparable { min(repeating(lhs, like: rhs), rhs) }
}

//==============================================================================
/// equal
/// Performs element-wise equality comparison and returns a
/// tensor of Bool values
@inlinable
public func equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>) -> Tensor<S,Bool>
where S: TensorShape, E: Equatable
{
    var result = Tensor<S, Bool>(lhs.shape)
    Context.currentQueue.equal(lhs, rhs, &result)
    return result
}

infix operator .== : ComparisonPrecedence

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

////==============================================================================
///// elementsAlmostEqual
///// Performs element-wise equality comparison within the tolerance range
///// and returns a tensor of Bool values
//@inlinable
//public func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T,
//                                   tolerance: T.Element) -> T.BoolView where
//    S: TensorShape, E: SignedNumeric & Comparable
//{
//    Context.currentQueue.elementsAlmostEqual(lhs, rhs, tolerance: tolerance)
//}
//
//public extension Platform {
//    @inlinable
//    func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T,
//                                tolerance: T.Element) -> T.BoolView
//        where S: TensorShape, E: SignedNumeric & Comparable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.elementsAlmostEqual(read(lhs), read(rhs),
//                                         tolerance, &resultBuffer)
//        return result
//    }
//}
//
//public extension TensorView where Element: SignedNumeric & Comparable {
//    @inlinable
//    func elementsAlmostEqual(_ rhs: Self, tolerance: Element) -> BoolView {
//        Context.currentQueue.elementsAlmostEqual(self, rhs, tolerance: tolerance)
//    }
//}
//
////==============================================================================
///// notEqual
///// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
///// values.
//@inlinable
//public func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView
//    where S: TensorShape, E: Equatable
//{
//    Context.currentQueue.notEqual(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView
//        where S: TensorShape, E: Equatable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.notEqual(read(lhs), read(rhs), &resultBuffer)
//        return result
//    }
//}
//
//infix operator .!= : ComparisonPrecedence
//
//public extension TensorView where Element: Equatable {
//    @inlinable
//    static func .!=(_ lhs: Self, _ rhs: Self) -> BoolView { notEqual(lhs, rhs) }
//}
//
////==============================================================================
///// greater
///// Computes `lhs .> rhs` element-wise and returns a tensor of Bool values
//@inlinable
//public func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//    S: TensorShape, E: Comparable
//{
//    Context.currentQueue.greater(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//        S: TensorShape, E: Comparable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.greater(read(lhs), read(rhs), &resultBuffer)
//        return result
//    }
//}
//
//infix operator .> : ComparisonPrecedence
//
//public extension TensorView where Element: Comparable {
//    @inlinable
//    static func .>(_ lhs: Self, _ rhs: Self) -> BoolView { greater(lhs, rhs) }
//}
//
////==============================================================================
///// greaterOrEqual
///// Computes `lhs .>= rhs` element-wise and returns a tensor of Bool values
//@inlinable
//public func greaterOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//    S: TensorShape, E: Comparable
//{
//    Context.currentQueue.greaterOrEqual(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func greaterOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//        S: TensorShape, E: Comparable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.greaterOrEqual(read(lhs), read(rhs), &resultBuffer)
//        return result
//    }
//}
//
//infix operator .>= : ComparisonPrecedence
//
//public extension TensorView where Element: Comparable {
//    @inlinable
//    static func .>=(_ lhs: Self, _ rhs: Self) -> BoolView {
//        greaterOrEqual(lhs, rhs)
//    }
//}
//
////==============================================================================
///// less
///// Computes `lhs .< rhs` element-wise and returns a tensor of Bool values
//@inlinable
//public func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//    S: TensorShape, E: Comparable
//{
//    Context.currentQueue.less(lhs, rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//        S: TensorShape, E: Comparable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.less(read(lhs), read(rhs), &resultBuffer)
//        return result
//    }
//}
//
//infix operator .< : ComparisonPrecedence
//
//public extension TensorView where Element: Comparable {
//    @inlinable
//    static func .<(_ lhs: Self, _ rhs: Self) -> BoolView { less(lhs, rhs) }
//}
//
////==============================================================================
///// lessOrEqual
///// Computes `lhs .<= rhs` element-wise and returns a tensor of Bool values
//@inlinable
//public func lessOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//    S: TensorShape, E: Comparable
//{
//    Context.currentQueue.lessOrEqual(lhs, rhs)
//}
//
//@inlinable
//public func lessOrEqual<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
//    where S: TensorShape, E: Comparable
//{
//    lessOrEqual(lhs, T(repeating: rhs, like: lhs))
//}
//
//@inlinable
//public func lessOrEqual<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
//    where S: TensorShape, E: Comparable
//{
//    lessOrEqual(T(repeating: lhs, like: rhs), rhs)
//}
//
//public extension Platform {
//    @inlinable
//    func lessOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
//        S: TensorShape, E: Comparable
//    {
//        assert(lhs.bounds == rhs.bounds, _messageTensorExtentsMismatch)
//        var result = lhs.createBoolTensor()
//        var resultBuffer = write(&result)
//        currentQueue.lessOrEqual(read(lhs), read(rhs), &resultBuffer)
//        return result
//    }
//
//    @inlinable
//    func lessOrEqual<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
//        where S: TensorShape, E: Comparable
//    {
//        lessOrEqual(lhs, T(repeating: rhs, like: lhs))
//    }
//
//    @inlinable
//    func lessOrEqual<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
//        where S: TensorShape, E: Comparable
//    {
//        lessOrEqual(T(repeating: lhs, like: rhs), rhs)
//    }
//}
//
//infix operator .<= : ComparisonPrecedence
//
//public extension TensorView where Element: Comparable {
//    @inlinable
//    static func .<=(_ lhs: Self, _ rhs: Self) -> BoolView {
//        lessOrEqual(lhs, rhs)
//    }
//
//    @inlinable
//    static func .<=(_ lhs: Self, _ rhs: Element) -> BoolView {
//        lessOrEqual(lhs, rhs)
//    }
//
//    @inlinable
//    static func .<=(_ lhs: Element, _ rhs: Self) -> BoolView {
//        lessOrEqual(lhs, rhs)
//    }
//}
