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
    _ config: PoolingConfiguration<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E: Numeric {
    // var status: cudaError_t
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pool(config, x, &out)
      return
    }
    diagnostic(.queueGpu, "pool(\(x.name))", categories: .queueGpu)

    // constants
    var zero = E.zero
    var one = E.one

    cudaCheck(
      cudnnPoolingForward(
        cudnn.handle,
        config.pooling.desc,
        // alpha
        &one,
        // xDesc
        config.xDesc.desc,
        // x
        x.deviceRead(using: self),
        // beta
        &zero,
        // yDesc
        config.outDesc.desc,
        // y
        out.deviceReadWrite(using: self)
      )
    )
  }

}  // CudaQueue

//==============================================================================
/// CudaPoolingConfiguration
///
public final class CudaPoolingConfiguration<Shape: TensorShape, E: StorageElement> {
  // properties
  public let pooling: PoolingDescriptor<Shape>
  public let xDesc: TensorDescriptor<Shape, E>

  // out
  public let outDesc: TensorDescriptor<Shape, E>
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
    self.init(
      x: x, windowSize: Shape(windowSize),
      strides: Shape(strides), padding: padding, mode: mode)
  }

  @inlinable public convenience init(
    x: Tensor<Shape, E>,
    windowSize: Shape.Tuple,
    strides: Shape.Tuple,
    padding: Shape.Tuple,
    mode: PoolingMode
  ) {
    self.init(
      x: x, windowSize: Shape(windowSize),
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
    let pad = padding == .valid ? Shape.zero : windowSize / 2
    // if `pad` is .valid then size `x` must be >= `windowSize`
    assert(
      padding == .same
        || {
          for i in 0..<Shape.rank {
            if windowSize[i] > x.shape[i] { return false }
          }
          return true
        }(), "with `.valid` padding, the input size `x` must be >= the windowSize")

    self.init(x: x, windowSize: windowSize, strides: strides, padding: pad, mode: mode)
  }

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:mode:
  @inlinable public init(
    x: Tensor<Shape, E>,
    windowSize: Shape,
    strides: Shape,
    padding: Shape,
    mode: PoolingMode
  ) {
    pooling = PoolingDescriptor(
      mode: mode,
      nan: .propagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor indenting dimensions with 1 as needed
    xDesc = TensorDescriptor(x)

    // get the output shape
    var shape32 = [Int32](repeating: 0, count: xDesc.rank)
    cudaCheck(
      cudnnGetPoolingNdForwardOutputDim(
        pooling.desc,
        xDesc.desc,
        Int32(xDesc.rank),
        &shape32
      )
    )

    // cudnn insists on ranks being 4 to 8, so skip leading 1s for lower ranks
    var s = Shape.zero
    let base = xDesc.rank > Shape.rank ? xDesc.rank - Shape.rank : 0
    for i in 0..<Shape.rank {
      s[i] = Int(shape32[base + i])
    }
    outShape = s

    // create output descriptor
    outOrder = x.order
    outDesc = TensorDescriptor(Tensor<Shape, E>(shape: outShape, order: outOrder))
  }

  @inlinable public func createOutput() -> Tensor<Shape, E> {
    Tensor<Shape, E>(shape: outShape, order: outOrder)
  }
}

//==============================================================================
/// PoolingDescriptor
public final class PoolingDescriptor<Shape: TensorShape> {
  // properties
  public let desc: cudnnPoolingDescriptor_t

  // initializers
  @inlinable public init(
    mode: PoolingMode,
    nan: NanPropagation,
    windowSize: Shape,
    padding: Shape,
    strides: Shape
  ) {
    // create the descriptor
    var temp: cudnnPoolingDescriptor_t!
    cudaCheck(cudnnCreatePoolingDescriptor(&temp))
    desc = temp

    // initialize
    cudaCheck(
      cudnnSetPoolingNdDescriptor(
        desc,
        mode.cudnn,
        nan.cudnn,
        Int32(Shape.rank),
        windowSize.asInt32,
        padding.asInt32,
        strides.asInt32
      )
    )
  }

  deinit {
    cudaCheck(cudnnDestroyLRNDescriptor(desc))
  }
}
