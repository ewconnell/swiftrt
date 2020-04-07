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
import Foundation

//==============================================================================
/// concat
/// - Parameter tensors: array of tensors whose elements will be joined
/// - Parameter axis: dimension to append the elements
@inlinable public func concat<S,E>(
    _ tensors: [Tensor<S,E>],
    alongAxis axis: Int = 0
) -> Tensor<S,E> where S: TensorShape
{
    assert(tensors.count > 1)
    // create result with joined shape
    let joinedShape = tensors[0].shape
        .joined(with: tensors[1...].map { $0.shape }, alongAxis: axis)

    var result = Tensor<S,E>(joinedShape)
    var lower = S.zero
    for tensor in tensors {
        copy(from: tensor, to: &result[lower, lower &+ tensor.shape])
        lower[axis] += tensor.shape[axis]
    }
    return result
}

public extension Tensor {
    @inlinable func concat(_ others: Self..., alongAxis axis: Int = 0) -> Self {
        guard others.count > 1 else { return self }
        return SwiftRT.concat([self] + others, alongAxis: axis)
    }
}

//==============================================================================
/// copy
/// copies the elements from `source` to `destination`
/// - Parameter source: tensor to be copied
/// - Parameter destination: the tensor where the result will be written
@inlinable public func copy<S,E>(
    from source: Tensor<S,E>,
    to destination: inout Tensor<S,E>
) where S: TensorShape
{
    Context.currentQueue.copy(from: source, to: &destination)
}

//==============================================================================
/// delayQueue
/// adds a time delay into the current queue for testing purposes``
/// - Parameter interval: the number of seconds to delay
@inlinable
public func delayQueue(atLeast interval: TimeInterval) {
    Context.platform.delayQueue(atLeast: interval)
}

public extension Platform {
    @inlinable
    func delayQueue(atLeast interval: TimeInterval) {
        currentQueue.delay(interval)
    }
}

//==============================================================================
// initializer extensions
public extension Platform {
    @inlinable
    func fill<T>(randomUniform x: inout T,
                 lowerBound: T.Element,
                 upperBound: T.Element,
                 seed: RandomSeed)
        where S: TensorShape, T.Element: BinaryFloatingPoint
    {
        var buffer = write(&x)
        currentQueue.fill(randomUniform: &buffer, lowerBound, upperBound, seed)
    }

    
    //-------------------------------------
    @inlinable
    func fill<T>(randomNormal x: inout T,
                 mean: T.Element,
                 standardDeviation: T.Element,
                 seed: RandomSeed)
        where S: TensorShape, T.Element: BinaryFloatingPoint
    {
        var buffer = write(&x)
        currentQueue.fill(randomNormal: &buffer, mean, standardDeviation, seed)
    }

    @inlinable
    func fill<T>(randomNormal x: inout T,
                 mean: T,
                 standardDeviation: T,
                 seed: RandomSeed)
        where S: TensorShape, T.Element: BinaryFloatingPoint
    {
        var buffer = write(&x)
        currentQueue.fill(randomNormal: &buffer, read(mean),
                          read(standardDeviation), seed)
    }

    //-------------------------------------
    @inlinable
    func fill<T>(randomTruncatedNormal x: inout T,
                    mean: T.Element, standardDeviation: T.Element,
                    seed: RandomSeed)
        where S: TensorShape, T.Element: BinaryFloatingPoint
    {
        var buffer = write(&x)
        currentQueue.fill(randomTruncatedNormal: &buffer, mean,
                          standardDeviation, seed)
    }

    @inlinable
    func fill<T>(randomTruncatedNormal x: inout T,
                 mean: T, standardDeviation: T,
                 seed: RandomSeed)
        where S: TensorShape, T.Element: BinaryFloatingPoint
    {
        var buffer = write(&x)
        currentQueue.fill(randomTruncatedNormal: &buffer, read(mean),
                          read(standardDeviation), seed)
    }
}

//==============================================================================
/// fill<T>(result:value:
/// fills the view with the specified value
@inlinable
public func fill<T>(_ result: inout T, with element: T.Element)
    where S: TensorShape
{
    Context.platform.fill(&result, with: element)
}

@inlinable
public func fill<T, R>(_ result: inout T, with range: R) where
    S: TensorShape,
    R: StridedRangeExpression & Collection, R.Element == T.Element
{
    Context.platform.fill(&result, with: range)
}

public extension Platform {
    @inlinable
    func fill<T>(_ result: inout T, with element: T.Element)
        where S: TensorShape
    {
        var resultBuffer = write(&result)
        currentQueue.fill(&resultBuffer, with: element)
    }
    
    /// fill(result:with range:
    /// fills the tensor with values formed by the specified range
    @inlinable
    func fill<T, R>(_ result: inout T, with range: R) where
        S: TensorShape,
        R: StridedRangeExpression & Collection, R.Element == T.Element
    {
        assert(result.count == range.stridedRange.count)
        var resultBuffer = write(&result)
        currentQueue.fill(&resultBuffer, with: range)
    }
}

public extension Tensor {
    /// filled
    /// creates a tensor shaped like Self and fills on device
    /// - Parameter element: the element value used to fill the tensor
    @inlinable func filled(with element: Element) -> Self {
        var result = createDense()
        fill(&result, with: element)
        return result
    }
    
    /// creates a tensor shaped like Self and fills on device
    /// - Parameter range: the range of values used to fill the tensor
    @inlinable func filled<R>(with range: R) -> Self
        where R: StridedRangeExpression & Collection, R.Element == Element
    {
        var result = createDense()
        fill(&result, with: range)
        return result
    }
}

//==============================================================================
/// fillWithIndex
/// a convenience function to fill the tensor with index values from
/// `0..<count`. If a different range is desired, use `fill(with range:`
public extension Platform {
    @inlinable
    func fillWithIndex<T>(_ result: inout T)
        where S: TensorShape, T.Element: RangeBound
    {
        let count = T.Element(exactly: result.count)!
        let range = StridedRange(from: 0, to: count, by: 1)
        fill(&result, with: range)
    }
}

public extension TensorView where Element: RangeBound {
    @inlinable
    func filledWithIndex() -> Self {
        var result = createDense()
        let count = Element(exactly: result.count)!
        let range = StridedRange(from: 0, to: count, by: 1)
        fill(&result, with: range)
        return result
    }
}

//==============================================================================
/// replace<T>(x:with:result:
/// fills the view with the specified value
@inlinable
public func replace<T>(x: T, with y: T, where condition: T.BoolView) -> T
    where S: TensorShape
{
    Context.platform.replace(x, with: y, where: condition)
}

public extension Platform {
    @inlinable
    func replace<T>(_ x: T, with y: T, where condition: T.BoolView) -> T
        where S: TensorShape
    {
        var (result, resultBuffer) = createResult(like: x)
        currentQueue.replace(read(x), read(y), read(condition), &resultBuffer)
        return result
    }
}

public extension TensorView where Element: Comparable {
    @inlinable
    func replacing(with y: Self, where condition: BoolView) -> Self {
        replace(x: self, with: y, where: condition)
    }
    
    @inlinable
    func replacing(with value: Element, where condition: BoolView) -> Self {
        replacing(with: Self(repeating: value, like: self), where: condition)
    }
}
