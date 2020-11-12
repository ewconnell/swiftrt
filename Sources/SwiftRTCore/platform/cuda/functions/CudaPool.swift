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
/// PoolingDescriptor
public final class PoolingDescriptor {
  /// the cudnn descriptor
  public let desc: cudnnPoolingDescriptor_t

  //----------------------------------------------------------------------------
  /// init(op:nan:windowSize:padding:strides:
  @inlinable public init<Shape: TensorShape>(
    op: PoolingOp,
    nan: NanPropagation,
    windowSize: Shape,
    padding: Shape,
    strides: Shape
  ) {
    var temp: cudnnPoolingDescriptor_t!
    cudaCheck(cudnnCreatePoolingDescriptor(&temp))
    desc = temp

    // cudnn doesn't support rank 1, so bump it up to 2
    if Shape.rank == 1 {
      cudaCheck(
        cudnnSetPoolingNdDescriptor(
          desc,
          op.cudnn,
          nan.cudnn,
          Int32(Shape.rank + 1),
          [1] + windowSize.asInt32,
          [0] + padding.asInt32,
          [1] + strides.asInt32
        )
      )
    } else {
      cudaCheck(
        cudnnSetPoolingNdDescriptor(
          desc,
          op.cudnn,
          nan.cudnn,
          Int32(Shape.rank),
          windowSize.asInt32,
          padding.asInt32,
          strides.asInt32
        )
      )
    }
  }

  deinit {
    cudaCheck(cudnnDestroyPoolingDescriptor(desc))
  }
}

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
  //--------------------------------------------------------------------------
  // Scalar Element
  @inlinable public func pool<Shape, E>(
    _ config: PoolingConfig<Shape, E>,
    _ x: Tensor<Shape, E>,
    _ out: inout Tensor<Shape, E>
  ) where E: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pool(config, x, &out)
      return
    }
    diagnostic(.queueGpu, "pool(\(x.name))", categories: .queueGpu)

    var zero = E.zero
    var one = E.one
    let status = cudnnPoolingForward(
      cudnn.handle,
      config.pooling.desc,
      // alpha
      &one,
      // xDesc
      config.x.desc,
      // x
      x.deviceRead(using: self),
      // beta
      &zero,
      // yDesc
      config.out.desc,
      // y
      out.deviceReadWrite(using: self)
    )
    cpuFallback(status) { $0.cpu_pool(config, x, &out) }
  }

  //--------------------------------------------------------------------------
  // Vector Element
  @inlinable public func pool<Shape, E>(
    _ config: PoolingConfig<Shape, E>,
    _ x: Tensor<Shape, E>,
    _ out: inout Tensor<Shape, E>
  ) where E: VectorElement, E.Scalar: Numeric {
    assert(out.isContiguous, _messageElementsMustBeContiguous)
    guard useGpu else {
      cpu_pool(config, x, &out)
      return
    }
    diagnostic(.queueGpu, "pool(\(x.name))", categories: .queueGpu)

    var zero = E.Scalar.zero
    var one = E.Scalar.one
    let status = cudnnPoolingForward(
      cudnn.handle,
      config.pooling.desc,
      // alpha
      &one,
      // xDesc
      config.x.desc,
      // x
      x.deviceRead(using: self),
      // beta
      &zero,
      // yDesc
      config.out.desc,
      // y
      out.deviceReadWrite(using: self)
    )
    cpuFallback(status) { $0.cpu_pool(config, x, &out) }
  }
}  // CudaQueue

//==============================================================================
/// CudaPoolingConfig
///
public final class CudaPoolingConfig<Shape: TensorShape, Element: StorageElement> {
  // properties
  public let pooling: PoolingDescriptor
  public let x: TensorDescriptor
  public let out: TensorDescriptor
  public let shape: Shape

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:op:
  @inlinable public init(
    x tensor: Tensor<Shape, Element>,
    windowSize: Shape,
    strides: Shape = Shape.one,
    padding: Shape = Shape.zero,
    op: PoolingOp
  ) {
    assert(
      op == .averagePadding || padding <= windowSize / 2,
      "padding cannot exceed half the window size")
    assert(windowSize <= tensor.shape &+ padding, "windowSize must be <= size x + padding")

    pooling = PoolingDescriptor(
      op: op,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor
    x = TensorDescriptor(tensor)

    // compute the output shape
    shape = 1 &+ (tensor.shape &+ 2 &* padding &- windowSize) / strides

    // create the output descriptor
    out = TensorDescriptor(Tensor<Shape, Element>(shape: shape, order: tensor.order))
  }

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:op:
  @inlinable public init(
    batch: Tensor<Shape, Element>,
    windowSize: Shape.M1,
    strides: Shape.M1 = Shape.M1.one,
    padding: Shape.M1 = Shape.M1.zero,
    op: PoolingOp
  ) {
    // get the shape of a single item
    var itemShape = Shape.M1.zero
    for i in 0..<Shape.M1.rank {
      itemShape[i] = batch.shape[i + 1]
    }
    assert(windowSize <= itemShape &+ padding, "windowSize must be <= size x + padding")

    pooling = PoolingDescriptor(
      op: op,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor indenting dimensions with 1 as needed
    x = TensorDescriptor(batch: batch)

    // compute the batch shape
    let outItemShape = 1 &+ (itemShape &+ 2 &* padding &- windowSize) / strides
    var batchShape = batch.shape
    for i in 0..<Shape.M1.rank {
      batchShape[i + 1] = outItemShape[i]
    }
    shape = batchShape

    // create output descriptor
    out = TensorDescriptor(batch: Tensor<Shape, Element>(shape: shape, order: batch.order))
  }
}
