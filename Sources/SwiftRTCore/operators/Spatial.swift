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
  public typealias PoolingConfiguration<S, E> = CudaPoolingConfiguration<S, E>
  where S: TensorShape, E: StorageElement
#else
  public typealias PoolingConfiguration<S, E> = CpuPoolingConfiguration<S, E>
  where S: TensorShape, E: StorageElement
#endif

//==============================================================================
/// PoolingMode
public enum PoolingMode: Int, Codable {
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
/// pool(_:window:strides:padding:mode:
/// computes the absolute value of `x`
/// - Parameters:
///  - x: input
///  - window: the size of the pooling window
///  - strides: the sliding window strides
///  - padding: the padding type
///  - mode: the pooling mode
///
/// - Returns: the pooled result
///
@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: S,
  strides: S,
  padding: S,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  let config = PoolingConfiguration(
    x: x, windowSize: windowSize, strides: strides, padding: padding, mode: mode)

  var out = config.createOutput()
  currentQueue.pool(config, x, &out)
  return out
}

//--------------------------------------
// using Int for simple symetrical cases
@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: Int,
  strides: Int = 1,
  padding: Int = 0,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  return pool(
    x, windowSize: S(repeating: windowSize), strides: S(repeating: strides), 
    padding: S(repeating: padding), mode: mode)
}

//--------------------------------------
// using tuples
@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: S.Tuple,
  strides: S.Tuple,
  padding: S.Tuple,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  return pool(x, windowSize: S(windowSize), strides: S(strides), padding: S(padding), mode: mode)
}

//--------------------------------------
// using enum for padding
@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: S,
  strides: S,
  padding: Padding,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  // create the pooling configuration
  let config = PoolingConfiguration(
    x: x, windowSize: windowSize, strides: strides, padding: padding, mode: mode)
  var out = config.createOutput()
  currentQueue.pool(config, x, &out)
  return out
}

@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: Int,
  strides: Int = 1,
  padding: Padding,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  return pool(x, windowSize: S(repeating: windowSize), 
    strides: S(repeating: strides), padding: padding, mode: mode)
}

@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  windowSize: S.Tuple,
  strides: S.Tuple = S.oneTuple,
  padding: Padding,
  mode: PoolingMode
) -> Tensor<S, E> where E: Numeric {
  return pool(x, windowSize: S(windowSize), strides: S(strides), padding: padding, mode: mode)
}
