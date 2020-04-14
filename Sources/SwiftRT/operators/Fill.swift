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
        result[lower, lower &+ tensor.shape] = tensor
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
@inlinable public func delayQueue(atLeast interval: TimeInterval) {
    Context.currentQueue.delay(interval)
}

//==============================================================================
// initializer extensions
@inlinable func fill<S,E>(
    randomUniform x: inout Tensor<S,E>,
    from lower: E,
    to upper: E,
    seed: RandomSeed
) where S: TensorShape, E: BinaryFloatingPoint
{
    Context.currentQueue.fill(randomUniform: &x, lower, upper, seed)
}

//-------------------------------------
@inlinable func fill<S,E>(
    randomNormal x: inout Tensor<S,E>,
    mean: E,
    standardDeviation: E,
    seed: RandomSeed
)
    where S: TensorShape, E: BinaryFloatingPoint
{
    Context.currentQueue.fill(randomNormal: &x, mean, standardDeviation, seed)
}

@inlinable func fill<S,E>(
    randomNormal x: inout Tensor<S,E>,
    mean: Tensor<S,E>,
    standardDeviation: Tensor<S,E>,
    seed: RandomSeed)
    where S: TensorShape, E: BinaryFloatingPoint
{
    Context.currentQueue.fill(randomNormal: &x, mean, standardDeviation, seed)
}

//-------------------------------------
@inlinable func fill<S,E>(
    randomTruncatedNormal x: inout Tensor<S,E>,
    mean: E, standardDeviation: E,
    seed: RandomSeed
) where S: TensorShape, E: BinaryFloatingPoint
{
    Context.currentQueue
        .fill(randomTruncatedNormal: &x, mean, standardDeviation, seed)
}

@inlinable func fill<S,E>(
    randomTruncatedNormal x: inout Tensor<S,E>,
    mean: Tensor<S,E>,
    standardDeviation: Tensor<S,E>,
    seed: RandomSeed
) where S: TensorShape, E: BinaryFloatingPoint
{
    Context.currentQueue
        .fill(randomTruncatedNormal: &x, mean, standardDeviation, seed)
}

//==============================================================================
/// fill<S,E>(x:value:
/// fills the view with the specified value
@inlinable public func fill<S,E>(_ x: inout Tensor<S,E>, with element: E)
    where S: TensorShape
{
    Context.currentQueue.fill(&x, with: element)
}

@inlinable
public func fill<S,E,B>(_ x: inout Tensor<S,E>, with range: Range<B>)
    where S: TensorShape, E: Numeric,
    B: SignedInteger, B.Stride: SignedInteger
{
    Context.currentQueue.fill(&x, with: range)
}

//==============================================================================
/// fillWithIndex
/// a convenience function to fill the tensor with index values from
/// `0..<count`. If a different range is desired, use `fill(with range:`
@inlinable func fillWithIndex<S,E>(_ x: inout Tensor<S,E>)
    where S: TensorShape, E: Comparable & Numeric
{
    fill(&x, with: 0..<x.count)
}

//==============================================================================
/// replace(x:with:result:
/// fills the view with the specified value
@inlinable public func replace<S,E>(
    x: Tensor<S,E>,
    with y: Tensor<S,E>,
    where condition: Tensor<S,Bool>
) -> Tensor<S,E> where S: TensorShape
{
    var result = Tensor(like: x)
    Context.currentQueue.replace(x, y, condition, &result)
    return result
}

public extension Tensor where Element: Comparable {
    @inlinable func replacing(
        with y: Self,
        where condition: Tensor<Shape,Bool>
    ) -> Self {
        replace(x: self, with: y, where: condition)
    }
    
    @inlinable func replacing(
        with value: Element,
        where condition: Tensor<Shape,Bool>
    ) -> Self {
        replacing(with: repeating(value, like: self), where: condition)
    }
}
