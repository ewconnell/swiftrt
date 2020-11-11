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
public final class PoolingDescriptor<Shape: TensorShape> {
  /// the cudnn descriptor
  public let desc: cudnnPoolingDescriptor_t

  //----------------------------------------------------------------------------
  /// init(op:nan:windowSize:padding:strides:
  @inlinable public init(
    op: PoolingOp,
    nan: NanPropagation,
    windowSize: Shape,
    padding: Shape,
    strides: Shape
  ) {
    var temp: cudnnPoolingDescriptor_t!
    cudaCheck(cudnnCreatePoolingDescriptor(&temp))
    desc = temp

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

  deinit {
    cudaCheck(cudnnDestroyPoolingDescriptor(desc))
  }
}

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension CudaQueue {
  //--------------------------------------------------------------------------
  // Scalar Element
  @inlinable public func pool<Config, Shape, E>(
    _ config: Config,
    _ x: Tensor<Shape, E>,
    _ out: inout Tensor<Shape, E>
  ) where Config: CudaPoolingConfigProtocol, E: Numeric {
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
      config.poolingDesc,
      // alpha
      &one,
      // xDesc
      config.xTensorDescriptor,
      // x
      x.deviceRead(using: self),
      // beta
      &zero,
      // yDesc
      config.outTensorDescriptor,
      // y
      out.deviceReadWrite(using: self)
    )
    cpuFallback(status) { $0.cpu_pool(config, x, &out) }
  }

  //--------------------------------------------------------------------------
  // Vector Element
  @inlinable public func pool<Config, Shape, E>(
    _ config: Config,
    _ x: Tensor<Shape, E>,
    _ out: inout Tensor<Shape, E>
  ) where Config: CudaPoolingConfigProtocol, E: VectorElement, E.Scalar: Numeric {
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
      config.poolingDesc,
      // alpha
      &one,
      // xDesc
      config.xTensorDescriptor,
      // x
      x.deviceRead(using: self),
      // beta
      &zero,
      // yDesc
      config.outTensorDescriptor,
      // y
      out.deviceReadWrite(using: self)
    )
    cpuFallback(status) { $0.cpu_pool(config, x, &out) }
  }
}  // CudaQueue

//==============================================================================
/// CudaPoolingConfigProtocol
public protocol CudaPoolingConfigProtocol: PoolingConfigProtocol {
  var poolingDesc: cudnnPoolingDescriptor_t { get }
  var xTensorDescriptor: cudnnTensorDescriptor_t { get }
  var outTensorDescriptor: cudnnTensorDescriptor_t { get }
}

//==============================================================================
/// CudaPoolingConfig
///
public final class CudaPoolingConfig<Shape, Element>: CudaPoolingConfigProtocol
where Shape: TensorShape, Element: StorageElement {
  // properties
  public let pooling: PoolingDescriptor<Shape>
  public let xDesc: TensorDescriptor
  public let outDesc: TensorDescriptor
  public let outShape: Shape

  @inlinable public var poolingDesc: cudnnPoolingDescriptor_t { pooling.desc }
  @inlinable public var xTensorDescriptor: cudnnTensorDescriptor_t { xDesc.desc }
  @inlinable public var outTensorDescriptor: cudnnTensorDescriptor_t { outDesc.desc }

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:op:
  @inlinable public init(
    x: Tensor<Shape, Element>,
    windowSize: Shape,
    strides: Shape = Shape.one,
    padding: Shape = Shape.zero,
    op: PoolingOp
  ) {
    assert(windowSize <= x.shape &+ padding, "windowSize must be <= size x + padding")

    pooling = PoolingDescriptor(
      op: op,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor
    xDesc = TensorDescriptor(x)

    // compute the output shape
    outShape = 1 &+ (x.shape &+ 2 &* padding &- windowSize) / strides

    // create the output descriptor
    outDesc = TensorDescriptor(Tensor<Shape, Element>(shape: outShape, order: x.order))
  }
}

//==============================================================================
/// CudaBatchPoolingConfig
///
public final class CudaBatchPoolingConfig<Shape, Element>: CudaPoolingConfigProtocol
where Shape: TensorShape, Element: StorageElement {
  public let pooling: PoolingDescriptor<Shape.M1>
  public let xDesc: TensorDescriptor
  public let outDesc: TensorDescriptor
  public let outShape: Shape

  @inlinable public var poolingDesc: cudnnPoolingDescriptor_t { pooling.desc }
  @inlinable public var xTensorDescriptor: cudnnTensorDescriptor_t { xDesc.desc }
  @inlinable public var outTensorDescriptor: cudnnTensorDescriptor_t { outDesc.desc }

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
      itemShape[i] = batch.shape[i+1]
    }
    assert(windowSize <= itemShape &+ padding, "windowSize must be <= size x + padding")

    pooling = PoolingDescriptor(
      op: op,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor indenting dimensions with 1 as needed
    xDesc = TensorDescriptor(batch: batch)

    // compute the batch shape
    let outItemShape = 1 &+ (itemShape &+ 2 &* padding &- windowSize) / strides
    var batchShape = batch.shape
    for i in 0..<Shape.M1.rank {
      batchShape[i+1] = outItemShape[i]
    }
    outShape = batchShape

    // create output descriptor
    outDesc = TensorDescriptor(batch: Tensor<Shape, Element>(shape: outShape, order: batch.order))
  }
}
