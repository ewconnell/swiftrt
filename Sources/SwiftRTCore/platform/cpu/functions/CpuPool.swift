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

//==============================================================================
public final class CpuPoolingConfiguration<Shape: TensorShape, E: StorageElement> {

  public let outOrder: Order
  public let outShape: Shape

  //----------------------------------------------------------------------------
  // Tuple helpers
  @inlinable public convenience init(
    x: Tensor<Shape, E>,
    windowSize: Shape.Tuple,
    strides: Shape.Tuple,
    padding: Padding,
    mode: PoolingMode
  ) {
    self.init(x: x, windowSize: Shape(windowSize), 
      strides: Shape(strides), padding: padding, mode: mode)
  }

  @inlinable public convenience init(
    x: Tensor<Shape, E>,
    windowSize: Shape.Tuple,
    strides: Shape.Tuple,
    padding: Shape.Tuple,
    mode: PoolingMode
  ) {
    self.init(x: x, windowSize: Shape(windowSize), 
      strides: Shape(strides), padding: Shape(padding), mode: mode)
  }

  //----------------------------------------------------------------------------
  // Padding version for TF compatibility
  // It converts to correct numeric padding and then delegates
  @inlinable public convenience init(
    x: Tensor<Shape, E>,
    windowSize: Shape,
    strides: Shape,
    padding: Padding,
    mode: PoolingMode
  ) {
    let pad = Shape.zero
    self.init(x: x, windowSize: windowSize, strides: strides, padding: pad, mode: mode)
  }

  //----------------------------------------------------------------------------
  @inlinable public init(
    x: Tensor<Shape, E>,
    windowSize: Shape,
    strides: Shape,
    padding: Shape,
    mode: PoolingMode
  ) {
    outOrder = x.order
    outShape = x.shape
  }

  @inlinable public func createOutput() -> Tensor<Shape, E> {
    Tensor<Shape, E>(shape: outShape, order: outOrder)
  }
}
