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
#if canImport(SwiftRTCuda)
  public typealias PoolingConfig<S, E> = CudaPoolingConfig<S, E>
  where S: TensorShape, E: StorageElement

  public typealias BatchPoolingConfig<S, E> = CudaBatchPoolingConfig<S, E>
  where S: TensorShape, E: StorageElement

#else
  public typealias PoolingConfig<S, E> = CpuPoolingConfig<S, E>
  where S: TensorShape, E: StorageElement
#endif

//==============================================================================
/// PoolingConfigProtocol
public protocol PoolingConfigProtocol {
  associatedtype Shape: TensorShape

  var outShape: Shape { get }
}

//==============================================================================
/// PoolingOp
public enum PoolingOp: Int, Codable {
  /// averages elements within the pooling window excluding padding
  case average
  /// averages elements within the pooling window including padding
  case averagePadding
  /// The maximum value inside the pooling window is used
  case max
  /// The maximum value inside the pooling window is used via a deterministic algorithm
  case maxDeterministic
}

//==============================================================================
/// pool(x:window:strides:padding:op:
/// computes the absolute value of `x`
/// - Parameters:
///  - x: input
///  - window: the size of the pooling window
///  - strides: the sliding window strides
///  - padding: the padding type
///  - op: the pooling operation type
///
/// - Returns: the pooled result
///
@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  windowSize: S,
  strides: S = S.one,
  padding: S = S.zero,
  op: PoolingOp
) -> Tensor<S, E> where E: Numeric {
  let config = PoolingConfig(
    x: x, windowSize: windowSize,
    strides: strides, padding: padding, op: op)

  var out = Tensor<S, E>(shape: config.outShape, order: x.order)
  currentQueue.pool(config, x, &out)
  return out
}

@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  windowSize: S.Tuple,
  strides: S.Tuple = S.oneTuple,
  padding: S.Tuple = S.zeroTuple,
  op: PoolingOp
) -> Tensor<S, E> where E: Numeric {
  return pool(
    x: x, windowSize: S(windowSize), strides: S(strides),
    padding: S(padding), op: op)
}

//--------------------------------------
// vector element
@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  windowSize: S,
  strides: S = S.one,
  padding: S = S.zero,
  op: PoolingOp
) -> Tensor<S, E> where E: VectorElement, E.Scalar: Numeric {
  let config = PoolingConfig(
    x: x, windowSize: windowSize,
    strides: strides, padding: padding, op: op)

  var out = Tensor<S, E>(shape: config.outShape, order: x.order)
  currentQueue.pool(config, x, &out)
  return out
}

@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  windowSize: S.Tuple,
  strides: S.Tuple = S.oneTuple,
  padding: S.Tuple = S.zeroTuple,
  op: PoolingOp
) -> Tensor<S, E> where E: VectorElement, E.Scalar: Numeric {
  return pool(
    x: x, windowSize: S(windowSize), strides: S(strides),
    padding: S(padding), op: op)
}

//--------------------------------------
// batched version
@inlinable public func pool<S, E>(
  batch: Tensor<S, E>,
  windowSize: S.M1,
  strides: S.M1 = S.M1.one,
  padding: S.M1 = S.M1.zero,
  op: PoolingOp
) -> Tensor<S, E> where E: Numeric {
  let config = BatchPoolingConfig(
    batch: batch, windowSize: windowSize,
    strides: strides, padding: padding, op: op)

  var out = Tensor<S, E>(shape: config.outShape, order: batch.order)
  currentQueue.pool(config, batch, &out)
  return out
}

@inlinable public func pool<S, E>(
  batch: Tensor<S, E>,
  windowSize: S.M1.Tuple,
  strides: S.M1.Tuple = S.M1.oneTuple,
  padding: S.M1.Tuple = S.M1.zeroTuple,
  op: PoolingOp
) -> Tensor<S, E> where E: Numeric {
  return pool(
    batch: batch, windowSize: S.M1(windowSize),
    strides: S.M1(strides), padding: S.M1(padding), op: op)
}

//--------------------------------------
// batched vector element version
@inlinable public func pool<S, E>(
  batch: Tensor<S, E>,
  windowSize: S.M1,
  strides: S.M1 = S.M1.one,
  padding: S.M1 = S.M1.zero,
  op: PoolingOp
) -> Tensor<S, E> where E: VectorElement, E.Scalar: Numeric {
  let config = BatchPoolingConfig(
    batch: batch, windowSize: windowSize,
    strides: strides, padding: padding, op: op)

  var out = Tensor<S, E>(shape: config.outShape, order: batch.order)
  currentQueue.pool(config, batch, &out)
  return out
}

@inlinable public func pool<S, E>(
  batch: Tensor<S, E>,
  windowSize: S.M1.Tuple,
  strides: S.M1.Tuple = S.M1.oneTuple,
  padding: S.M1.Tuple = S.M1.zeroTuple,
  op: PoolingOp
) -> Tensor<S, E> where E: VectorElement, E.Scalar: Numeric {
  return pool(
    batch: batch, windowSize: S.M1(windowSize),
    strides: S.M1(strides), padding: S.M1(padding), op: op)
}
