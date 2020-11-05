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

import Foundation
import Numerics

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CpuQueue {
  //--------------------------------------------------------------------------
  @inlinable public func pool<S, E>(
    _ config: PoolingConfiguration<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E: Numeric {
    cpu_pool(config, x, &out)
  }

}

//==============================================================================
// Cpu device queue function implementations
extension DeviceQueue {
  //--------------------------------------------------------------------------
  @inlinable public func cpu_pool<S, E>(
    _ config: PoolingConfiguration<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E: Numeric {
    fatalError("cpu_pool not implemented yet")
    // diagnostic(.queueCpu, "pool(\(x.name))", categories: .queueCpu)

  }
}
