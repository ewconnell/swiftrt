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
extension Tensor where TensorElement.Value == Bool {
    /// Computes `lhs .&& rhs` element-wise and returns a tensor of Bool values
    @inlinable public static func .&&(_ lhs: Self, _ rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor(like: lhs)
        Context.currentQueue.and(lhs, rhs, &result)
        return result
    }
}

//==============================================================================
/// or
public extension Tensor where TensorElement.Value == Bool {
    /// Computes `lhs .|| rhs` element-wise and returns a tensor of Bool values
    @inlinable static func .||(_ lhs: Self, _ rhs: Self) -> Self {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor(like: lhs)
        Context.currentQueue.or(lhs, rhs, &result)
        return result
    }
}

//==============================================================================
/// max
/// Computes the element-wise maximum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result

// tensor tensor
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func max<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.max(lhs, rhs, &result)
    return result
}

@derivative(of: max)
@usableFromInline func _vjpMax<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
where S: TensorShape, E.Value: DifferentiableNumeric & Comparable {
    (value: max(lhs, rhs), {
        var resultTrue = Tensor(like: lhs)
        var resultFalse = Tensor(like: lhs)
        Context.currentQueue.vjpMax(lhs, rhs, $0, &resultTrue, &resultFalse)
        return (resultTrue, resultFalse)
    })
}

//--------------------------------
// tensor Element
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func max<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    var result = Tensor(like: lhs)
    Context.currentQueue.max(lhs, rhs, &result)
    return result
}

@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func max<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Int
) -> Tensor<S,E> where E.Value: Comparable & Numeric {
    max(lhs, E.Value(exactly: rhs)!)
}

@derivative(of: max)
@usableFromInline func _vjpMax<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, E.Value))
where S: TensorShape, E.Value: Comparable & Numeric & DifferentiableNumeric {
    // Dan
    fatalError()
}

@derivative(of: max, wrt: lhs)
@usableFromInline func _vjpMax<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E.Value: Comparable & Numeric & DifferentiableNumeric {
    // Dan
    fatalError()
}

//--------------------------------
// Element tensor
// delegate to reverse
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func max<S,E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    max(rhs, lhs)
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor where TensorElement.Value: Comparable {
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func max(_ lhs: Self, _ rhs: Self) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func max(_ lhs: Self, _ rhs: TensorElement.Value) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func max(_ lhs: TensorElement.Value, _ rhs: Self) -> Self {
        SwiftRTCore.max(lhs, rhs)
    }
}

//==============================================================================
/// min
/// Computes the element-wise minimum of two tensors
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor
/// - Returns: result
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func min<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
    var result = Tensor(like: lhs)
    Context.currentQueue.min(lhs, rhs, &result)
    return result
}

@derivative(of: min)
@usableFromInline func _vjpMin<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Tensor<S,E>
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
    where S: TensorShape, E.Value: DifferentiableNumeric & Comparable
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
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func min<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    var result = Tensor(like: lhs)
    Context.currentQueue.min(lhs, rhs, &result)
    return result
}

@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func min<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: Int
) -> Tensor<S,E> where E.Value: Comparable & Numeric {
    min(lhs, E.Value(exactly: rhs)!)
}

@derivative(of: min)
@usableFromInline func _vjpMin<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, E.Value)
) where S: TensorShape, E.Value: Comparable & DifferentiableNumeric {
    // Dan
    fatalError()
}

@derivative(of: min, wrt: lhs)
@usableFromInline func _vjpMin<S,E>(
    _ lhs: Tensor<S,E>,
    _ rhs: E.Value
) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> Tensor<S,E>)
where S: TensorShape, E.Value: Comparable & Numeric & DifferentiableNumeric {
    // Dan
    fatalError()
}

//--------------------------------
// Element tensor
@differentiable(where E.Value: DifferentiableNumeric)
@inlinable public func min<S,E>(
    _ lhs: E.Value,
    _ rhs: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape, E.Value: Comparable {
    min(rhs, lhs)
}

// These are added to disambiguate from Swift max when writing
// a TensorView extension
public extension Tensor where TensorElement.Value: Comparable {
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func min(_ lhs: Self, _ rhs: Self) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func min(_ lhs: Self, _ rhs: TensorElement.Value) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }

    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable func min(_ lhs: TensorElement.Value, _ rhs: Self) -> Self {
        SwiftRTCore.min(lhs, rhs)
    }
}

//==============================================================================
/// equal
extension Tensor: Equatable where TensorElement.Value: Equatable {
    /// Performs element-wise equality comparison and returns a
    /// tensor of Bool values
    @inlinable public static func .== (
        _ lhs: Self,
        _ rhs: Self
    ) -> Tensor<Shape, Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape, Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.equal(lhs, rhs, &result)
        return result
    }

    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor
    /// - Returns: `true` if the tensors are equal
    @inlinable public static func == (lhs: Self, rhs: Self) -> Bool {
        // the bounds must match or they are not equal
        guard lhs.shape == rhs.shape else { return false }

        // if lhs is an alias for rhs, then they match
        if lhs.storage === rhs.storage && lhs.storageBase == rhs.storageBase {
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
) -> Tensor<S,Bool> where S: TensorShape, E.Value: SignedNumeric & Comparable {
    assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
    var result = Tensor<S,Bool>(shape: lhs.shape, order: lhs.order)
    Context.currentQueue.elementsAlmostEqual(lhs, rhs, tolerance, &result)
    return result
}

public extension Tensor where TensorElement.Value: SignedNumeric & Comparable {
    @inlinable func elementsAlmostEqual(
        _ rhs: Self,
        tolerance: TensorElement.Value
    ) -> Tensor<Shape,Bool> {
        SwiftRTCore.elementsAlmostEqual(self, rhs, tolerance: tolerance)
    }
}

//==============================================================================
/// notEqual
/// Computes `lhs != rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public extension Tensor where TensorElement.Value: Equatable {
    @inlinable static func .!=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape, Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.notEqual(lhs, rhs, &result)
        return result
    }
}

//==============================================================================
/// greater
/// Computes `lhs .> rhs` element-wise and returns a tensor of Bool values
public extension Tensor where TensorElement.Value: Comparable {
    @inlinable static func .>(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.greater(lhs, rhs, &result)
        return result
    }

    @inlinable static func .>(_ lhs: Self, _ rhs: Element) -> Tensor<Shape,Bool> {
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.greater(lhs, rhs, &result)
        return result
    }
}

@inlinable public func .><S,E>(_ lhs: Tensor<S,Complex<E>>, _ rhs: E) -> Tensor<S,Bool> {
    lhs .> Complex<E>(rhs)
}

//==============================================================================
/// greaterOrEqual
public extension Tensor where TensorElement.Value: Comparable {
    /// Computes `lhs .>= rhs` element-wise and returns a tensor of Bool values
    @inlinable static func .>=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.greaterOrEqual(lhs, rhs, &result)
        return result
    }
    
    @inlinable static func .>=(_ lhs: Self, _ rhs: Element) -> Tensor<Shape,Bool> {
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.greaterOrEqual(lhs, rhs, &result)
        return result
    }
}

//==============================================================================
/// less
public extension Tensor where TensorElement.Value: Comparable {
    /// Computes `lhs .< rhs` element-wise and returns a tensor of Bool values
    @inlinable static func .<(_ lhs: Self, _ rhs: Self) -> Tensor<Shape, Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.less(lhs, rhs, &result)
        return result
    }
    
    @inlinable static func .<(_ lhs: Self, _ rhs: Element) -> Tensor<Shape,Bool> {
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.less(lhs, rhs, &result)
        return result
    }
}

//==============================================================================
/// lessOrEqual
public extension Tensor where TensorElement.Value: Comparable {
    /// Computes `lhs .<= rhs` element-wise and returns a tensor of Bool values
    @inlinable static func .<=(_ lhs: Self, _ rhs: Self) -> Tensor<Shape,Bool> {
        assert(lhs.shape == rhs.shape, _messageTensorShapeMismatch)
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.lessOrEqual(lhs, rhs, &result)
        return result
    }

    @inlinable static func .<=(_ lhs: Self, _ rhs: Element) -> Tensor<Shape,Bool> {
        var result = Tensor<Shape,Bool>(shape: lhs.shape, order: lhs.order)
        Context.currentQueue.lessOrEqual(lhs, rhs, &result)
        return result
    }
}
