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
}

//==============================================================================
/// pool(_:size:strides:pad:mode:
/// computes the absolute value of `x`
/// - Parameters:
///  - x: input
///  - size: the size of the pooling window
///  - strides: the sliding window strides
///  - pad: the padding type
///  - mode: the pooling mode
///
/// - Returns: the pooled result
///
@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  size: S.Tuple,
  strides: S.Tuple,
  pad: Padding = .valid,
  mode: PoolingMode = .average
) -> Tensor<S, E> where E: Numeric {
  return pool(x: x, size: S(size), strides: S(strides), pad: pad, mode: mode)
}

@inlinable public func pool<S, E>(
  x: Tensor<S, E>,
  size: S,
  strides: S,
  pad: Padding = .valid,
  mode: PoolingMode = .average
) -> Tensor<S, E> where E: Numeric {
  // create the pooling configuration
  let config = PoolingConfiguration<S, E>(
    x: x, size: size, strides: strides, pad: pad, mode: mode)

  var out = config.createOutput()
  currentQueue.pool(config, x, &out)
  return out
}

@inlinable public func pool<S, E>(
  config: PoolingConfiguration<S, E>,
  x: Tensor<S, E>,
  out: inout Tensor<S, E>
) -> Tensor<S, E> where E: Numeric {
  currentQueue.pool(config, x, &out)
  return out
}

