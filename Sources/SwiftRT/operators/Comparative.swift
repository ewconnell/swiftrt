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
public extension ComputePlatform {
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
public extension ComputePlatform {
    func and<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element == Bool
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.and(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    func and<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        and(lhs, T(repeating: rhs, like: lhs))
    }
    
    func and<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        and(T(repeating: lhs, like: rhs), rhs)
    }
}

infix operator .&& : LogicalConjunctionPrecedence

public extension TensorView where Element == Bool {
    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.and(lhs, rhs)
    }
    
    @inlinable
    static func .&&(_ lhs: Self, _ rhs: Element) -> BoolView {
        Current.platform.and(lhs, rhs)
    }
    
    @inlinable
    static func .&&(_ lhs: Element, _ rhs: Self) -> BoolView {
        Current.platform.and(lhs, rhs)
    }
}

//==============================================================================
/// or
/// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
public extension ComputePlatform {
    func or<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element == Bool
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.or(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    func or<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        or(lhs, T(repeating: rhs, like: lhs))
    }
    
    func or<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element == Bool
    {
        or(T(repeating: lhs, like: rhs), rhs)
    }
}

infix operator .|| : LogicalConjunctionPrecedence

public extension TensorView where Element == Bool {
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.or(lhs, rhs)
    }
    
    @inlinable
    static func .||(_ lhs: Self, _ rhs: Element) -> BoolView {
        Current.platform.or(lhs, rhs)
    }
    
    @inlinable
    static func .||(_ lhs: Element, _ rhs: Self) -> BoolView {
        Current.platform.or(lhs, rhs)
    }
}

//==============================================================================
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
public extension ComputePlatform {
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

public extension TensorView {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.max(lhs, rhs) }

    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T, _ rhs: T.Element) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.max(lhs, rhs) }

    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ lhs: T.Element, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.max(lhs, rhs) }
}

//--------------------------------------
// derivative functions
public extension ComputePlatform {
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
public extension ComputePlatform {
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

public extension TensorView {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }

    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T, _ rhs: T.Element) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }

    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ lhs: T.Element, _ rhs: T) -> T where
        T: TensorView, T.Element: Comparable { Current.platform.min(lhs, rhs) }
}

//--------------------------------------
// derivative functions
public extension ComputePlatform {
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
public extension ComputePlatform {
    func equal<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

public extension TensorView where Element: Equatable {
    @inlinable
    static func .== (_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.equal(lhs, rhs)
    }
    
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
public extension ComputePlatform {
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
    func elementsAlmostEqual(_ other: Self, tolerance: Element) -> BoolView {
        Current.platform.elementsAlmostEqual(self, other, tolerance: tolerance)
    }
}

//==============================================================================
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public extension ComputePlatform {
    func notEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where T: TensorView {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.notEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

public extension TensorView where Element: Equatable {
    @inlinable
    static func .!=(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.notEqual(lhs, rhs)
    }
}

//==============================================================================
/// greater
/// Computes `lhs .> rhs` element-wise and returns a tensor of Bool values
public extension ComputePlatform {
    func greater<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.greater(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

public extension TensorView where Element: Comparable {
    @inlinable
    static func .>(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.greater(lhs, rhs)
    }
}

//==============================================================================
/// greaterOrEqual
/// Computes `lhs .>= rhs` element-wise and returns a tensor of Bool values
public extension ComputePlatform {
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

public extension TensorView where Element: Comparable {
    @inlinable
    static func .>=(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.greaterOrEqual(lhs, rhs)
    }
}

//==============================================================================
/// less
/// Computes `lhs .< rhs` element-wise and returns a tensor of Bool values
public extension ComputePlatform {
    func less<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.less(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

public extension TensorView where Element: Comparable {
    @inlinable
    static func .<(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.less(lhs, rhs)
    }
}

//==============================================================================
/// lessOrEqual
/// Computes `lhs .<= rhs` element-wise and returns a tensor of Bool values
public extension ComputePlatform {
    func lessOrEqual<T>(_ lhs: T, _ rhs: T) -> T.BoolView where
        T: TensorView, T.Element: Comparable
    {
        assert(lhs.extents == rhs.extents, _messageTensorExtentsMismatch)
        var result = lhs.createBoolTensor()
        currentQueue.lessOrEqual(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
    
    func lessOrEqual<T>(_ lhs: T, _ rhs: T.Element) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    {
        lessOrEqual(lhs, T(repeating: rhs, like: lhs))
    }
    
    func lessOrEqual<T>(_ lhs: T.Element, _ rhs: T) -> T.BoolView
        where T: TensorView, T.Element: Comparable
    {
        lessOrEqual(T(repeating: lhs, like: rhs), rhs)
    }
}

public extension TensorView where Element: Comparable {
    @inlinable
    static func .<=(_ lhs: Self, _ rhs: Self) -> BoolView {
        Current.platform.lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Self, _ rhs: Element) -> BoolView {
        Current.platform.lessOrEqual(lhs, rhs)
    }

    @inlinable
    static func .<=(_ lhs: Element, _ rhs: Self) -> BoolView {
        Current.platform.lessOrEqual(lhs, rhs)
    }
}
