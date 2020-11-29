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
/// all(x:axes:
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The out extent along the specified
/// axes will be 1. Rank is not reduced.
/// - Parameters:
///  - x: value tensor
/// - Returns: result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable public func all<S>(
  _ x: Tensor<S, Bool>,
  axes: [Int]? = nil
) -> Tensor<S, Bool> {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, Bool>(shape: shape)
  currentQueue.all(x, &out)
  return out
}

/// - Parameter along: the axes to operate on
/// - Returns: a new tensor containing the out
extension Tensor where TensorElement == Bool {
  @inlinable public func all(axes: [Int]? = nil) -> Self {
    SwiftRTCore.all(self, axes: axes)
  }

  @inlinable public func all(axes: Int...) -> Self { all(axes: axes) }
}

//==============================================================================
/// any(x:axes:
/// Returns `true` if any value is equal to `true` along the specified
/// axes. Otherwise returns `false`. The out extent along the specified
/// axes will be 1. Rank is not reduced.
/// - Parameters:
///  - x: value tensor
/// - Returns: result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable public func any<S>(
  _ x: Tensor<S, Bool>,
  axes: [Int]? = nil
) -> Tensor<S, Bool> {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, Bool>(shape: shape)
  currentQueue.any(x, &out)
  return out
}

/// - Parameter axes: the axes to operate on
/// - Returns: a new tensor containing the out
extension Tensor where TensorElement == Bool {
  @inlinable public func any(axes: [Int]? = nil) -> Self {
    SwiftRTCore.any(self, axes: axes)
  }

  @inlinable public func any(axes: Int...) -> Self { any(axes: axes) }
}

//==============================================================================
/// sum(x:along:
/// Sums `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func sum<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.sum(x, &out)
  return out
}

extension Tensor where TensorElement.Value: Numeric {
  @inlinable public func sum(axes: [Int]? = nil) -> Self {
    SwiftRTCore.sum(self, axes: axes)
  }

  @inlinable public func sum(axes: Int...) -> Self { sum(axes: axes) }
}

//==============================================================================
/// mean(x:along:
/// mean of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func mean<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: AlgebraicField {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.mean(x, &out)
  return out
}

extension Tensor where TensorElement.Value: AlgebraicField {
  @inlinable public func mean(axes: [Int]? = nil) -> Self {
    SwiftRTCore.mean(self, axes: axes)
  }

  @inlinable public func mean(axes: Int...) -> Self { mean(axes: axes) }
}

//==============================================================================
/// prod(x:along:
/// prod of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func prod<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.prod(x, &out)
  return out
}

extension Tensor where TensorElement.Value: Numeric {
  @inlinable public func prod(axes: [Int]? = nil) -> Self {
    SwiftRTCore.prod(self, axes: axes)
  }

  @inlinable public func prod(axes: Int...) -> Self { prod(axes: axes) }
}

//==============================================================================
/// prodNonZeros(x:along:
/// product of non zero values of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func prodNonZeros<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: Numeric {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.prodNonZeros(x, &out)
  return out
}

extension Tensor where TensorElement.Value: Numeric {
  @inlinable public func prodNonZeros(axes: [Int]? = nil) -> Self {
    SwiftRTCore.prodNonZeros(self, axes: axes)
  }

  @inlinable public func prodNonZeros(axes: Int...) -> Self {
    prodNonZeros(axes: axes)
  }
}

//==============================================================================
/// min(x:along:
/// returns the minimum element value of `x` along the specified axes
/// TODO: add optional indices
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func min<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: Comparable & ComparableLimits {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.min(x, &out)
  return out
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  @inlinable public func min(axes: [Int]? = nil) -> Self {
    SwiftRTCore.min(self, axes: axes)
  }

  @inlinable public func min(axes: Int...) -> Self {
    min(axes: axes)
  }
}

//==============================================================================
/// max(x:along:
/// returns the maximum element value of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func max<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: Comparable & ComparableLimits {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.max(x, &out)
  return out
}

extension Tensor where TensorElement.Value: Comparable & ComparableLimits {
  @inlinable public func max(axes: [Int]? = nil) -> Self {
    SwiftRTCore.max(self, axes: axes)
  }

  @inlinable public func max(axes: Int...) -> Self {
    max(axes: axes)
  }
}

//==============================================================================
/// abssum(x:along:
/// Sums the absolute values of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable public func abssum<S, E>(
  _ x: Tensor<S, E>,
  axes: [Int]? = nil
) -> Tensor<S, E> where E.Value: SignedNumeric & Comparable {
  let shape = axes == nil ? S.one : x.reductionShape(along: axes!)
  var out = Tensor<S, E>(shape: shape)
  currentQueue.abssum(x, &out)
  return out
}

extension Tensor where TensorElement.Value: SignedNumeric & Comparable {
  @inlinable public func abssum(axes: [Int]? = nil) -> Self {
    SwiftRTCore.abssum(self, axes: axes)
  }

  @inlinable public func abssum(axes: Int...) -> Self {
    abssum(axes: axes)
  }
}
