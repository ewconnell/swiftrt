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
@inlinable
public func concat<T>(tensors: [T], alongAxis axis: Int = 0,
                      name: String? = nil) -> T where T: TensorView
{
    Platform.service.concat(tensors, alongAxis: axis, name)
}

public extension PlatformService {
    @inlinable
    func concat<T>(_ tensors: [T], alongAxis axis: Int = 0,
                          _ name: String? = nil) -> T where T: TensorView
    {
        assert(tensors.count > 1)
        // compute joined shape and create result buffer
        let joinedShape = tensors[0].shape
            .joined(with: tensors[1...].map { $0.shape }, alongAxis: axis)

        var (result, resultBuffer) = createResult(like: tensors[0],
                                                  with: joinedShape,
                                                  name: name)

        currentQueue.concat(tensors.map {read($0)},
                            along: axis, &resultBuffer)
        return result
    }
}

public extension TensorView {
    @inlinable
    func concat(_ others: Self..., alongAxis axis: Int = 0,
                name: String? = nil) -> Self
    {
        Platform.service.concat([self] + others, alongAxis: axis, name)
    }
}

//==============================================================================
/// copy
/// copies the elements from `view` to `result`
/// - Parameter from view: tensor to be copied
/// - Parameter to result: the tensor where the result will be written
@inlinable
public func copy<T>(from view: T, to result: inout T) where T: TensorView {
    Platform.service.copy(from: view, to: &result)
}

public extension PlatformService {
    @inlinable
    func copy<T>(from view: T, to result: inout T) where T: TensorView
    {
        var resultBuffer = write(&result)
        currentQueue.copy(from: read(view), to: &resultBuffer)
    }
}

//==============================================================================
/// delayQueue
/// adds a time delay into the current queue for testing purposes``
/// - Parameter interval: the number of seconds to delay
@inlinable
public func delayQueue(atLeast interval: TimeInterval) {
    Platform.service.delayQueue(atLeast: interval)
}

public extension PlatformService {
    @inlinable
    func delayQueue(atLeast interval: TimeInterval) {
        currentQueue.delay(interval)
    }
}

//==============================================================================
/// fill<T>(result:value:
/// fills the view with the specified value
@inlinable
public func fill<T>(_ result: inout T, with element: T.Element)
    where T: TensorView
{
    Platform.service.fill(&result, with: element)
}

@inlinable
public func fill<T, R>(_ result: inout T, with range: R) where
    T: TensorView,
    R: StridedRangeExpression & Collection, R.Bound == T.Element
{
    Platform.service.fill(&result, with: range)
}

public extension PlatformService {
    @inlinable
    func fill<T>(_ result: inout T, with element: T.Element)
        where T: TensorView
    {
        var resultBuffer = write(&result)
        currentQueue.fill(&resultBuffer, with: element)
    }
    
    /// fill(result:with range:
    /// fills the tensor with values formed by the specified range
    @inlinable
    func fill<T, R>(_ result: inout T, with range: R) where
        T: TensorView,
        R: StridedRangeExpression & Collection, R.Bound == T.Element
    {
        assert(result.count == range.stridedRange.count)
        var resultBuffer = write(&result)
        currentQueue.fill(&resultBuffer, with: range)
    }
}

public extension TensorView {
    /// filled
    /// creates a tensor shaped like Self and fills on device
    /// - Parameter element: the element value used to fill the tensor
    @inlinable
    func filled(with element: Element) -> Self {
        var result = createDense()
        fill(&result, with: element)
        return result
    }
    
    /// creates a tensor shaped like Self and fills on device
    /// - Parameter range: the range of values used to fill the tensor
    @inlinable
    func filled<R>(with range: R) -> Self
        where R: StridedRangeExpression & Collection, R.Bound == Element
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
public extension PlatformService {
    @inlinable
    func fillWithIndex<T>(_ result: inout T)
        where T: TensorView, T.Element: AnyNumeric & RangeBound
    {
        let range = StridedRange(from: 0, to: T.Element(any: result.count), by: 1)
        fill(&result, with: range)
    }
}

public extension TensorView where Element: AnyNumeric & RangeBound {
    @inlinable
    func filledWithIndex() -> Self {
        var result = createDense()
        let range = StridedRange(from: 0, to: Element(any: result.count), by: 1)
        fill(&result, with: range)
        return result
    }
}

//==============================================================================
/// replace<T>(x:with:result:
/// fills the view with the specified value
@inlinable
public func replace<T>(x: T, with y: T, where condition: T.BoolView) -> T
    where T: TensorView
{
    Platform.service.replace(x, with: y, where: condition)
}

public extension PlatformService {
    @inlinable
    func replace<T>(_ x: T, with y: T, where condition: T.BoolView) -> T
        where T: TensorView
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
