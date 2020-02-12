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
import Real

//==============================================================================
// utilities
extension ComputePlatform {
    @inlinable
    func _vjpMinMaxHelper<T>(
        _ x: T, _ y: T, v: T,
        op: @escaping (T.Element, T.Element) -> Bool) -> (T, T)
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        var resultTrue = x.createDense()
        var resultFalse = x.createDense()
        currentQueue.vjpMinMax(
            x: x, y: y, scale: v, op: op,
            resultTrue: &resultTrue, resultFalse: &resultFalse)
        return (resultTrue, resultFalse)
    }
}

//==============================================================================
/// and
/// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
@inlinable
public func and<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element == Bool
{
    Current.platform.and(lhs, rhs)
}

@inlinable
public func and<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
    where T: TensorView, T.Element == Bool
{
    Current.platform.and(lhs, rhs)
}

@inlinable
func and<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
    where T: TensorView, T.Element == Bool
{
    Current.platform.and(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func and<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element == Bool
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.and(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @inlinable
    func and<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        and(lhs, T(repeating: rhs, like: lhs))
    }
    
    @inlinable
    func and<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        and(T(repeating: lhs, like: rhs), rhs)
    }
}

infix operator .&& : LogicalConjunctionPrecedence

public extension TensorView where Element == Bool {
    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Self) -> BoolView { and(lhs, rhs) }
    
    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Element) -> BoolView { and(lhs, rhs) }
    
    @inlinable
    static func .&&(_ lhs: Element, _ rhs: Self) -> BoolView { and(lhs, rhs) }
}

//==============================================================================
/// or
/// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
@inlinable
public func or<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element == Bool
{
    Current.platform.or(lhs, rhs)
}

@inlinable
public func or<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
    where T: TensorView, T.Element == Bool
{
    Current.platform.or(lhs, rhs)
}

@inlinable
public func or<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
    where T: TensorView, T.Element == Bool
{
    Current.platform.or(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func or<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element == Bool
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.or(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @inlinable
    func or<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        or(lhs, T(repeating: rhs, like: lhs))
    }
    
    @inlinable
    func or<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        or(T(repeating: lhs, like: rhs), rhs)
    }
}

infix operator .|| : LogicalConjunctionPrecedence

public extension TensorView where Element == Bool {
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Self) -> BoolView { or(lhs, rhs) }
    
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Element) -> BoolView { or(lhs, rhs) }
    
    @inlinable
    static func .||(_ lhs: Element, _ rhs: Self) -> BoolView { or(lhs, rhs) }
}

//==============================================================================
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable
@differentiable(where T: DifferentiableTensorView)
public func max<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    Current.platform.max(lhs, rhs)
}

@inlinable
@differentiable(where T: DifferentiableTensorView)
public func max<T>(_ lhs: T, _ rhs: T.Element) -> T where
    T: TensorView, T.Element: Comparable
{
    Current.platform.max(lhs, rhs)
}

@inlinable
@differentiable(where T: DifferentiableTensorView)
public func max<T>(_ lhs: T.Element, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    Current.platform.max(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.max(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T, _ rhs: T.Element) -> T where
        T: TensorView, T.Element: Comparable
    {
        max(lhs, T(repeating: rhs, like: lhs))
    }
    
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T.Element, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable
    {
        max(T(repeating: lhs, like: rhs), rhs)
    }
}

//public extension TensorView {
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func max<T>(_ lhs: T, _ rhs: T) -> T where
//        T: TensorView, T.Element: Comparable { max(lhs, rhs) }
//
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func max<T>(_ lhs: T, _ rhs: T.Element) -> T where
//        T: TensorView, T.Element: Comparable { max(lhs, rhs) }
//
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func max<T>(_ lhs: T.Element, _ rhs: T) -> T where
//        T: TensorView, T.Element: Comparable { max(lhs, rhs) }
//}

//--------------------------------------
// derivative functions
extension ComputePlatform {
    @inlinable
    @derivative(of: max)
    func _vjpMax<T>(_ lhs: T, _ rhs: T)
        -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        return (value: max(lhs, rhs), {
            _vjpMinMaxHelper(lhs, rhs, v: $0, op: >=)
        })
    }
    
    @inlinable
    @derivative(of: max)
    func _vjpMax<T>(_ lhs: T, _ rhs: T.Element) ->
        (value: T, pullback: (T) -> (T, T.Element))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        let rhs = T(repeating: rhs, like: lhs)
        return (value: max(lhs, rhs), {
            let (lhsGrad, rhsGrad) = _vjpMinMaxHelper(lhs, rhs, v: $0, op: >=)
            return (lhsGrad, rhsGrad.sum().element)
        })
    }
    
    @inlinable
    @derivative(of: max)
    func _vjpMax<T>(_ lhs: T.Element, _ rhs: T) ->
        (value: T, pullback: (T) -> (T.Element, T))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        let lhs = T(repeating: lhs, like: rhs)
        return (value: max(lhs, rhs), {
            let (lhsGrad, rhsGrad) = _vjpMinMaxHelper(lhs, rhs, v: $0, op: >=)
            return (lhsGrad.sum().element, rhsGrad)
        })
    }
}

//==============================================================================
/// min
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@inlinable
@differentiable(where T: DifferentiableTensorView)
public func min<T>(_ lhs: T, _ rhs: T) -> T where
    T: TensorView, T.Element: Comparable
{
    Current.platform.min(lhs, rhs)
}

@inlinable
@differentiable(where T: DifferentiableTensorView)
public func min<T>(_ lhs: T, _ rhs: T.Element) -> T
    where T: TensorView, T.Element: Comparable
{
    Current.platform.min(lhs, rhs)
}

@inlinable
@differentiable(where T: DifferentiableTensorView)
public func min<T>(_ lhs: T.Element, _ rhs: T) -> T
    where T: TensorView, T.Element: Comparable
{
    Current.platform.min(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createDense()
        currentQueue.min(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T, _ rhs: T.Element) -> T
        where T: TensorView, T.Element: Comparable
    {
        min(lhs, T(repeating: rhs, like: lhs))
    }
    
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T.Element, _ rhs: T) -> T
        where T: TensorView, T.Element: Comparable
    {
        min(T(repeating: lhs, like: rhs), rhs)
    }
}

//public extension TensorView {
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func min<T>(_ lhs: T, _ rhs: T) -> T where
//        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }
//
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func min<T>(_ lhs: T, _ rhs: T.Element) -> T where
//        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }
//
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    func min<T>(_ lhs: T.Element, _ rhs: T) -> T where
//        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }
//}

//--------------------------------------
// derivative functions
extension ComputePlatform {
    @inlinable
    @derivative(of: min)
    func _vjpMin<T>(_ lhs: T, _ rhs: T)
        -> (value: T, pullback: (T) -> (T, T))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        return (value: min(lhs, rhs), {
            _vjpMinMaxHelper(lhs, rhs, v: $0, op: <=)
        })
    }
    
    @inlinable
    @derivative(of: min)
    func _vjpMin<T>(_ lhs: T, _ rhs: T.Element) ->
        (value: T, pullback: (T) -> (T, T.Element))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        let rhs = T(repeating: rhs, like: lhs)
        return (value: min(lhs, rhs), {
            let (lhsGrad, rhsGrad) = _vjpMinMaxHelper(lhs, rhs, v: $0, op: <=)
            return (lhsGrad, rhsGrad.sum().element)
        })
    }
    
    @inlinable
    @derivative(of: min)
    func _vjpMin<T>(_ lhs: T.Element, _ rhs: T) ->
        (value: T, pullback: (T) -> (T.Element, T))
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        let lhs = T(repeating: lhs, like: rhs)
        return (value: min(lhs, rhs), {
            let (lhsGrad, rhsGrad) = _vjpMinMaxHelper(lhs, rhs, v: $0, op: <=)
            return (lhsGrad.sum().element, rhsGrad)
        })
    }
}

//==============================================================================
/// equal
/// Performs element-wise equality comparison and returns a
/// tensor of Bool values
@inlinable
public func equal<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
    Current.platform.equal(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func equal<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

infix operator .== : ComparisonPrecedence

public extension TensorView where Element: Equatable {
    @inlinable
    static func .== (_ lhs: Self, _ rhs: Self) -> BoolView { equal(lhs, rhs) }
    
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        // the extents must match or they are not equal
        guard lhs.extents == rhs.extents else { return false }
        
        // if lhs is an alias for rhs, then they match
        if lhs.tensorArray === rhs.tensorArray &&
            lhs.viewOffset == rhs.viewOffset { return true }
        
        // compare elements
        return (lhs .== rhs).all().element
    }
}

//==============================================================================
/// elementsAlmostEqual
/// Performs element-wise equality comparison within the tolerance range
/// and returns a tensor of Bool values
@inlinable
public func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T,
                            tolerance: T.Element) -> T.BoolView where
    T: TensorView, T.Element: SignedNumeric & Comparable
{
    Current.platform.elementsAlmostEqual(lhs, rhs, tolerance: tolerance)
}

extension ComputePlatform {
    @inlinable
    func elementsAlmostEqual<T>(_ lhs: T, _ rhs: T,
                                tolerance: T.Element) -> T.BoolView where
        T: TensorView, T.Element: SignedNumeric & Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.elementsAlmostEqual(lhs: lhs, rhs: rhs,
                                         tolerance: tolerance,
                                         result: &result)
        return result
    }
}

public extension TensorView where Element: SignedNumeric & Comparable {
    @inlinable
    func elementsAlmostEqual(_ other: Self, tolerance: Element) -> BoolView {
        Current.platform.elementsAlmostEqual(self, other, tolerance: tolerance)
    }
}

//==============================================================================
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
@inlinable
public func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
    Current.platform.notEqual(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.notEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

infix operator .!= : ComparisonPrecedence

public extension TensorView where Element: Equatable {
    @inlinable
    static func .!=(_ lhs: Self, _ rhs: Self) -> BoolView { notEqual(lhs, rhs) }
}

//==============================================================================
/// greater
/// Computes `lhs .> rhs` element-wise and returns a tensor of Bool values
@inlinable
public func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Comparable
{
    Current.platform.greater(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.greater(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

infix operator .> : ComparisonPrecedence

public extension TensorView where Element: Comparable {
    @inlinable
    static func .>(_ lhs: Self, _ rhs: Self) -> BoolView { greater(lhs, rhs) }
}

//==============================================================================
/// greaterOrEqual
/// Computes `lhs .>= rhs` element-wise and returns a tensor of Bool values
@inlinable
public func greaterOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Comparable
{
    Current.platform.greaterOrEqual(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func greaterOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.greaterOrEqual(lhs: lhs, rhs: rhs,
                                    result: &result)
        return result
    }
}

infix operator .>= : ComparisonPrecedence

public extension TensorView where Element: Comparable {
    @inlinable
    static func .>=(_ lhs: Self, _ rhs: Self) -> BoolView {
        greaterOrEqual(lhs, rhs)
    }
}

//==============================================================================
/// less
/// Computes `lhs .< rhs` element-wise and returns a tensor of Bool values
@inlinable
public func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Comparable
{
    Current.platform.less(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.less(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

infix operator .< : ComparisonPrecedence

public extension TensorView where Element: Comparable {
    @inlinable
    static func .<(_ lhs: Self, _ rhs: Self) -> BoolView { less(lhs, rhs) }
}

//==============================================================================
/// lessOrEqual
/// Computes `lhs .<= rhs` element-wise and returns a tensor of Bool values
@inlinable
public func lessOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
    T: TensorView, T.Element: Comparable
{
    Current.platform.lessOrEqual(lhs, rhs)
}

@inlinable
public func lessOrEqual<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
    where T: TensorView, T.Element: Comparable
{
    Current.platform.lessOrEqual(lhs, rhs)
}

@inlinable
public func lessOrEqual<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
    where T: TensorView, T.Element: Comparable
{
    Current.platform.lessOrEqual(lhs, rhs)
}

extension ComputePlatform {
    @inlinable
    func lessOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.lessOrEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    @inlinable
    func lessOrEqual<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    {
        lessOrEqual(lhs, T(repeating: rhs, like: lhs))
    }
    
    @inlinable
    func lessOrEqual<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    {
        lessOrEqual(T(repeating: lhs, like: rhs), rhs)
    }
}

infix operator .<= : ComparisonPrecedence

public extension TensorView where Element: Comparable {
    @inlinable
    static func .<=(_ lhs: Self, _ rhs: Self) -> BoolView {
        lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Self, _ rhs: Element) -> BoolView {
        lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Element, _ rhs: Self) -> BoolView {
        lessOrEqual(lhs, rhs)
    }
}
