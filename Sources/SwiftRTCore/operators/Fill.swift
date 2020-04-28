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
@differentiable(where E: DifferentiableElement)
@inlinable public func concat<S,E>(
    _ tensors: [Tensor<S,E>],
    alongAxis axis: Int = 0,
    into result: inout Tensor<S,E>
) where S: TensorShape
{
    assert(tensors.count > 1)
    assert(result.shape == joinedShape(tensors,axis),
           "result shape does not match expected shape")
    
    var lower = S.zero
    for tensor in tensors {
        result[lower, lower &+ tensor.shape] = tensor
        lower[axis] += tensor.shape[axis]
    }
}

@differentiable(where E: DifferentiableElement)
@inlinable public func concat<S,E>(
    _ tensors: [Tensor<S,E>],
    alongAxis axis: Int = 0
) -> Tensor<S,E> where S: TensorShape
{
    var result = withoutDerivative(at: Tensor<S,E>(joinedShape(tensors,axis)))
    concat(tensors, alongAxis: axis, into: &result)
    return result
}

public extension Tensor {
    @differentiable(where Element: DifferentiableElement)
    @inlinable func concat(_ others: Self..., alongAxis axis: Int = 0) -> Self {
        guard others.count > 1 else { return self }
        return SwiftRTCore.concat([self] + others, alongAxis: axis)
    }
}

@inlinable public func joinedShape<S,E>(
    _ tensors: [Tensor<S,E>],
    _ axis: Int
) -> S {
    var shape = tensors[0].shape
    for i in 1..<tensors.count {
        shape[axis] += tensors[i].shape[axis]
    }
    return shape
}

@derivative(of: concat)
func vjpConcat<S,E>(
    _ tensors: [Tensor<S,E>],
    axis: Int = 0,
    into result: inout Tensor<S,E>
) -> (value: (), pullback: (inout Tensor<S, E>.TangentVector)
        -> Array<Tensor<S, E>>.TangentVector)
where S: TensorShape
{
    fatalError()
//    let tensorCount = tensors.count
//    func pullback(_ resultTangent: inout Tensor<SR, E>.TangentVector)
//    -> Array<Tensor<S, E>>.TangentVector
//    {
//        // Fill `tensorTangents` with slices of `resultTangent` of shape
//        // `tensorShapes[0]`, `tensorShapes[1]`, etc.
//        var tensorTangents: [Tensor<S, E>] = []
//        var lower = SR.zero
//        var upper = resultTangent.shape
//        upper[axis] = 1
//        for _ in 0..<tensorCount {
//            let slice = Tensor<S,E>(squeezing: resultTangent[lower, upper],
//                                    axes: Shape1(axis))
//            tensorTangents.append(slice)
//            lower[axis] += 1
//            upper[axis] += 1
//        }
//
//        // Set `resultTangent` to zero.
//        // Note: We can't use `fill(_:with:)` because `resultTangent` aliases
//        // `tensorTangents`.
//        // TODO: track and fix
//        // Note: https://bugs.swift.org/browse/TF-1250 will allow us to make
//        // this pullback more efficient. How:
//        // - Set the wrt parameters and results to
//        //     @differentiable(wrt: (tensors), results: (result))
//        // - This makes `resultTangent` not be inout, so we don't need to set
//        //   it any more.
//        resultTangent = zeros(like: resultTangent)
//
//        return Array.DifferentiableView(tensorTangents)
//    }
//    return (stack(tensors, axis: axis, into: &result), pullback)
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
