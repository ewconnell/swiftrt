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
/// all(x:axis:
/// Returns `true` if all values are equal to `true` along the specified
/// axis. Otherwise returns `false`. The out extent along the specified
/// axis will be 1. Rank is not reduced.
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func all<S>(
  _ x: Tensor<S, Bool>,
  axis: Int? = nil
) -> Tensor<S, Bool> {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, Bool>(shape: shape)
  currentQueue.all(x, axis, &out)
  return out
}

@inlinable public func all<S>(
  _ x: Tensor<S, Bool>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, Bool> {
  Tensor(reshaping: all(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement == Bool {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func all(axis: Int? = nil) -> Self {
    SwiftRTCore.all(self, axis: axis)
  }
}

//==============================================================================
/// any(x:axis:
/// Returns `true` if any value is equal to `true` along the specified
/// axis. Otherwise returns `false`. The out extent along the specified
/// axis will be 1. Rank is not reduced.
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func any<S>(
  _ x: Tensor<S, Bool>,
  axis: Int? = nil
) -> Tensor<S, Bool> {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, Bool>(shape: shape)
  currentQueue.any(x, axis, &out)
  return out
}

@inlinable public func any<S>(
  _ x: Tensor<S, Bool>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, Bool> {
  Tensor(reshaping: any(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement == Bool {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func any(axis: Int? = nil) -> Self {
    SwiftRTCore.any(self, axis: axis)
  }
}

//==============================================================================
/// sum(x:axis:
/// Sums `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func sum<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.sum(x, axis, &out)
  return out
}

@inlinable public func sum<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: Numeric {
  Tensor(reshaping: sum(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where Element: Numeric {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func sum(axis: Int? = nil) -> Self {
    SwiftRTCore.sum(self, axis: axis)
  }
}

//==============================================================================
/// mean(x:axis:
/// mean of `x` along the specified axis
///
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func mean<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: AlgebraicField {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.mean(x, axis, &out)
  return out
}

@inlinable public func mean<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: AlgebraicField {
  Tensor(reshaping: mean(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: AlgebraicField {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func mean(axis: Int? = nil) -> Self {
    SwiftRTCore.mean(self, axis: axis)
  }
}

//==============================================================================
/// prod(x:axis:
/// prod of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func prod<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.prod(x, axis, &out)
  return out
}

@inlinable public func prod<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: Numeric {
  Tensor(reshaping: prod(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: Numeric {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func prod(axis: Int? = nil) -> Self {
    SwiftRTCore.prod(self, axis: axis)
  }
}

//==============================================================================
/// prodNonZeros(x:axis:
/// product of non zero values of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func prodNonZeros<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.prodNonZeros(x, axis, &out)
  return out
}

@inlinable public func prodNonZeros<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: Numeric {
  Tensor(reshaping: prodNonZeros(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: Numeric {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func prodNonZeros(axis: Int? = nil) -> Self {
    SwiftRTCore.prodNonZeros(self, axis: axis)
  }
}

//==============================================================================
/// min(x:axis:
/// returns the minimum element value of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func min<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: Comparable & ComparableLimits {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.min(x, axis, &out)
  return out
}

@inlinable public func min<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: Comparable & ComparableLimits {
  Tensor(reshaping: min(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func min(axis: Int? = nil) -> Self {
    SwiftRTCore.min(self, axis: axis)
  }
}

//==============================================================================
/// argmin(x:axis:
/// returns the minimum element value of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func argmin<S, E>(
  _ x: Tensor<S, E>,
  axis: Int = 0
) -> (index: Tensor<S, Int32>, value: Tensor<S, E>) where E.Value: Comparable & ComparableLimits {
  let shape = x.reductionShape(axis)
  var arg = Tensor<S, Int32>(shape: shape)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.argmin(x, axis, &arg, &out)
  return (arg, out)
}

@inlinable public func argmin<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> (index: Tensor<S.M1, Int32>, value: Tensor<S.M1, E>)
where E.Value: Comparable & ComparableLimits {
  let (a, v) = argmin(x, axis: axis)
  let shape = x.shape.minus(axis)
  return (
    Tensor(reshaping: a, to: shape, order: x.order),
    Tensor(reshaping: v, to: shape, order: x.order)
  )
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default is axis 0
  /// - Returns: a new tensor containing the out
  @inlinable public func argmin(axis: Int = 0) -> (index: Tensor<Shape, Int32>, value: Self) {
    SwiftRTCore.argmin(self, axis: axis)
  }

  /// - Parameters:
  ///  - squeezingAxis: the axis to operate on and remove.
  /// - Returns: a new tensor one rank lower containing the result
  @inlinable public func argmin(squeezingAxis axis: Int) -> (
    index: Tensor<Shape.M1, Int32>, value: Tensor<Shape.M1, TensorElement>
  ) {
    SwiftRTCore.argmin(self, squeezingAxis: axis)
  }
}

//==============================================================================
/// max(x:axis:
/// returns the maximum element value of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func max<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: Comparable & ComparableLimits {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.max(x, axis, &out)
  return out
}

@inlinable public func max<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: Comparable & ComparableLimits {
  Tensor(reshaping: max(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func max(axis: Int? = nil) -> Self {
    SwiftRTCore.max(self, axis: axis)
  }
}

//==============================================================================
/// argmax(x:axis:
/// returns the maximum element value of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func argmax<S, E>(
  _ x: Tensor<S, E>,
  axis: Int = 0
) -> (index: Tensor<S, Int32>, value: Tensor<S, E>) where E.Value: Comparable & ComparableLimits {
  let shape = x.reductionShape(axis)
  var arg = Tensor<S, Int32>(shape: shape)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.argmax(x, axis, &arg, &out)
  return (arg, out)
}

@inlinable public func argmax<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> (index: Tensor<S.M1, Int32>, value: Tensor<S.M1, E>)
where E.Value: Comparable & ComparableLimits {
  let (a, v) = argmax(x, axis: axis)
  let shape = x.shape.minus(axis)
  return (
    Tensor(reshaping: a, to: shape, order: x.order),
    Tensor(reshaping: v, to: shape, order: x.order)
  )
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  /// - Parameters:
  ///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
  /// - Returns: a new tensor containing the out
  @inlinable public func argmax(axis: Int = 0) -> (index: Tensor<Shape, Int32>, value: Self) {
    SwiftRTCore.argmax(self, axis: axis)
  }

  /// - Parameters:
  ///  - squeezingAxis: the axis to operate on and remove.
  /// - Returns: a new tensor one rank lower containing the result
  @inlinable public func argmax(squeezingAxis axis: Int) -> (
    index: Tensor<Shape.M1, Int32>, value: Tensor<Shape.M1, TensorElement>
  ) {
    SwiftRTCore.argmax(self, squeezingAxis: axis)
  }
}

//==============================================================================
/// abssum(x:axis:
/// Sums the absolute values of `x` along the specified axis
/// - Parameters:
///  - x: value tensor
///  - axis: the axis to operate on. Default `nil` reduces entire flattened tensor
/// - Returns: result
/// - Precondition: Each value in `axis` must be in the range `-rank..<rank`.
@inlinable public func abssum<S, E>(
  _ x: Tensor<S, E>,
  axis: Int? = nil
) -> Tensor<S, E> where E.Value: SignedNumeric & Comparable {
  let shape = axis == nil ? S.one : x.reductionShape(axis!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.abssum(x, axis, &out)
  return out
}

@inlinable public func abssum<S, E>(
  _ x: Tensor<S, E>,
  squeezingAxis axis: Int
) -> Tensor<S.M1, E> where E.Value: SignedNumeric & Comparable {
  Tensor(reshaping: abssum(x, axis: axis), to: x.shape.minus(axis), order: x.order)
}

extension Tensor where TensorElement.Value: SignedNumeric & Comparable {
  /// - Parameters:
  /// - Returns: a new tensor containing the out

  @inlinable public func abssum(axis: Int? = nil) -> Self {
    SwiftRTCore.abssum(self, axis: axis)
  }
}
