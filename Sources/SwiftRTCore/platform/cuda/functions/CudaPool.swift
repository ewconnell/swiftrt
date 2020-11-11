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
    cpuFallback(status) { $0.cpu_pool(config, x, &out) }
  }
}  // CudaQueue

//==============================================================================
/// CudaPoolingConfigProtocol
public protocol CudaPoolingConfigProtocol: PoolingConfigProtocol {
  var pooling: PoolingDescriptor<Shape> { get }
  var xDesc: TensorDescriptor<Shape, Element> { get }
  var outDesc: TensorDescriptor<Shape, Element> { get }
}

//==============================================================================
/// CudaPoolingConfig
///
public final class CudaPoolingConfig<Shape, Element> : CudaPoolingConfigProtocol
  where Shape: TensorShape, Element: StorageElement
{
  // properties
  public let pooling: PoolingDescriptor<Shape>
  public let xDesc: TensorDescriptor<Shape, Element>
  public let outDesc: TensorDescriptor<Shape, Element>
  public let outOrder: Order
  public let outShape: Shape

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:op:
  @inlinable public init(
    x: Tensor<Shape, Element>,
    windowSize: Shape,
    strides: Shape = Shape.one,
    padding: Shape = Shape.zero,
    op: PoolingOp
  ) {
    // if `pad` is .valid then size `x` must be >= `windowSize`
    assert(
      {
        let inputShape = x.shape &+ padding
        for i in 0..<Shape.rank {
          if windowSize[i] > inputShape[i] { return false }
        }
        return true
      }(), "input `x` plus `padding` must be >= the windowSize")

    pooling = PoolingDescriptor(
      op: op,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor indenting dimensions with 1 as needed
    xDesc = TensorDescriptor(x, false)

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
    outDesc = TensorDescriptor(Tensor<Shape, Element>(shape: outShape, order: outOrder), false)
  }

  @inlinable public func createOutput() -> Tensor<Shape, Element> {
    Tensor<Shape, Element>(shape: outShape, order: outOrder)
  }
}

//==============================================================================
/// CudaBatchPoolingConfig
///
public final class CudaBatchPoolingConfig<Shape, Element> : CudaPoolingConfigProtocol
  where Shape: TensorShape, Element: StorageElement
{
  public let pooling: PoolingDescriptor<Shape>
  public let xDesc: TensorDescriptor<Shape, Element>
  public let outDesc: TensorDescriptor<Shape, Element>
  public let outOrder: Order
  public let outShape: Shape

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:op:
  @inlinable public init(
    batch: Tensor<Shape, Element>,
    windowSize: Shape.M1,
    strides: Shape.M1 = Shape.M1.one,
    padding: Shape.M1 = Shape.M1.zero,
    op: PoolingOp
  ) {
    fatalError()
    // // if `pad` is .valid then size `x` must be >= `windowSize`
    // assert(
    //   {
    //     let inputShape = x.shape &+ padding
    //     for i in 0..<Shape.rank {
    //       if windowSize[i] > inputShape[i] { return false }
    //     }
    //     return true
    //   }(), "input `x` plus `padding` must be >= the windowSize")

    // pooling = PoolingDescriptor(
    //   op: op,
    //   nan: .noPropagate,
    //   windowSize: windowSize,
    //   padding: padding,
    //   strides: strides)

    // // create input descriptor indenting dimensions with 1 as needed
    // xDesc = TensorDescriptor(x, false)

    // // get the output shape
    // var shape32 = [Int32](repeating: 0, count: xDesc.rank)
    // cudaCheck(
    //   cudnnGetPoolingNdForwardOutputDim(
    //     pooling.desc,
    //     xDesc.desc,
    //     Int32(xDesc.rank),
    //     &shape32
    //   )
    // )

    // // cudnn insists on ranks being 4 to 8, so skip leading 1s for lower ranks
    // var s = Shape.zero
    // let base = xDesc.rank > Shape.rank ? xDesc.rank - Shape.rank : 0
    // for i in 0..<Shape.rank {
    //   s[i] = Int(shape32[base + i])
    // }
    // outShape = s

    // // create output descriptor
    // outOrder = x.order
    // outDesc = TensorDescriptor(Tensor<Shape, E>(shape: outShape, order: outOrder), false)
  }

  @inlinable public func createOutput() -> Tensor<Shape, Element> {
    Tensor<Shape, Element>(shape: outShape, order: outOrder)
  }
}
