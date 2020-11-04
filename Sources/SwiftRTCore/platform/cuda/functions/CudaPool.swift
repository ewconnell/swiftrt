//******************************************************************************
// Copyright 2020 Google LLC
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

import SwiftRTCuda

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
  //--------------------------------------------------------------------------
  @inlinable public func pool<S, E>(
    _ x: Tensor<S, E>,
    _ size: S,
    _ strides: S,
    _ pad: Padding,
    _ op: PoolingOp,
    _ out: inout Tensor<S, E>
  ) {
    var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pool(x, size, strides, pad, op, &out)
      return
    }

  }

}
