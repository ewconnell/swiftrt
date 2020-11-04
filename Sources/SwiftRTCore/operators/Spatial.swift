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

public enum PoolingOp {
  case average
  case max
}

//==============================================================================
/// pool(_:size:strides:pad:op:
/// computes the absolute value of `x`
/// - Parameters:
///  - x: input
///  - size: the size of the pooling window
///  - strides: the sliding window strides
///  - pad: the padding mode
///  - op: the pooling operation
///
/// - Returns: the pooled result
///
@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  size: S.Tuple,
  strides: S.Tuple,
  pad: Padding = .valid,
  op: PoolingOp = .average
) -> Tensor<S, E> {
  return pool(x, size: S(size), strides: S(strides), pad: pad, op: op)
}

@inlinable public func pool<S, E>(
  _ x: Tensor<S, E>,
  size: S,
  strides: S,
  pad: Padding = .valid,
  op: PoolingOp = .average
) -> Tensor<S, E> {
  var out = Tensor(like: x)
  currentQueue.pool(x, size, strides, pad, op, &out)
  return out
}
