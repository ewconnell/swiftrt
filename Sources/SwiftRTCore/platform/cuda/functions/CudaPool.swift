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
  // Scalar Element version
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

  //--------------------------------------------------------------------------
  // Vector Element version
  @inlinable public func pool<S, E>(
    _ config: PoolingConfiguration<S, E>,
    _ x: Tensor<S, E>,
    _ out: inout Tensor<S, E>
  ) where E: VectorElement, E.Scalar: Numeric {
    // var status: cudaError_t
    // assert(out.isContiguous, _messageElementsMustBeContiguous)
    // guard useGpu else {
    //   cpu_pool(config, x, &out)
    //   return
    // }
    // diagnostic(.queueGpu, "pool(\(x.name))", categories: .queueGpu)

    // // constants
    // var zero = E.zero
    // var one = E.one

    // cudaCheck(
    //   cudnnPoolingForward(
    //     cudnn.handle,
    //     config.pooling.desc,
    //     // alpha
    //     &one,
    //     // xDesc
    //     config.xDesc.desc,
    //     // x
    //     x.deviceRead(using: self),
    //     // beta
    //     &zero,
    //     // yDesc
    //     config.outDesc.desc,
    //     // y
    //     out.deviceReadWrite(using: self)
    //   )
    // )
  }

}  // CudaQueue

//==============================================================================
/// CudaPoolingConfiguration
///
public final class CudaPoolingConfiguration<S: TensorShape, E: StorageElement> {
  // properties
  public let pooling: PoolingDescriptor<S>
  public let xDesc: TensorDescriptor<S, E>

  // out
  public let outDesc: TensorDescriptor<S, E>
  public let outOrder: Order
  public let outShape: S

  //----------------------------------------------------------------------------
  // Tuple helpers
  @inlinable public convenience init(
    x: Tensor<S, E>,
    windowSize: S.Tuple,
    strides: S.Tuple = S.oneTuple,
    padding: Padding,
    mode: PoolingMode
  ) {
    self.init(
      x: x, windowSize: S(windowSize),
      strides: S(strides), padding: padding, mode: mode)
  }

  @inlinable public convenience init(
    x: Tensor<S, E>,
    windowSize: S.Tuple,
    strides: S.Tuple = S.oneTuple,
    padding: S.Tuple = S.zeroTuple,
    mode: PoolingMode
  ) {
    self.init(
      x: x, windowSize: S(windowSize),
      strides: S(strides), padding: S(padding), mode: mode)
  }

  //------------------------------------
  // batch version
  @inlinable public convenience init(
    batch: Tensor<S, E>,
    windowSize: S.Tuple,
    strides: S.Tuple = S.oneTuple,
    padding: Padding,
    mode: PoolingMode
  ) {
    self.init(
      batch: batch, windowSize: S(windowSize),
      strides: S(strides), padding: padding, mode: mode)
  }

  @inlinable public convenience init(
    batch: Tensor<S, E>,
    windowSize: S.Tuple,
    strides: S.Tuple = S.oneTuple,
    padding: S.Tuple = S.zeroTuple,
    mode: PoolingMode
  ) {
    self.init(
      batch: batch, windowSize: S(windowSize),
      strides: S(strides), padding: S(padding), mode: mode)
  }

  //----------------------------------------------------------------------------
  // Padding version for TF compatibility
  // It converts to correct numeric padding and then delegates
  @inlinable public convenience init(
    x: Tensor<S, E>,
    windowSize: S,
    strides: S = S.one,
    padding: Padding,
    mode: PoolingMode
  ) {
    let pad = padding == .valid ? S.zero : windowSize / 2
    self.init(x: x, windowSize: windowSize, strides: strides, padding: pad, mode: mode)
  }

  //------------------------------------
  // batch version
  @inlinable public convenience init(
    batch: Tensor<S, E>,
    windowSize: S,
    strides: S = S.one,
    padding: Padding,
    mode: PoolingMode
  ) {
    let pad = padding == .valid ? S.zero : windowSize / 2
    self.init(batch: batch, windowSize: windowSize, strides: strides, padding: pad, mode: mode)
  }

  //----------------------------------------------------------------------------
  /// init(x:windowSize:strides:padding:mode:
  @inlinable public convenience init(
    x: Tensor<S, E>,
    windowSize: S,
    strides: S = S.one,
    padding: S = S.zero,
    mode: PoolingMode
  ) {
    self.init(
      x: x, windowSize: windowSize, strides: strides,
      padding: padding, mode: mode, isBatch: false)
  }

  /// init(batch:windowSize:strides:padding:mode:
  @inlinable public convenience init(
    batch: Tensor<S, E>,
    windowSize: S,
    strides: S = S.one,
    padding: S = S.zero,
    mode: PoolingMode
  ) {
    self.init(
      x: batch, windowSize: windowSize, strides: strides,
      padding: padding, mode: mode, isBatch: true)
  }

  //----------------------------------------------------------------------------
  @usableFromInline init(
    x: Tensor<S, E>,
    windowSize: S,
    strides: S,
    padding: S,
    mode: PoolingMode,
    isBatch: Bool
  ) {
    // if `pad` is .valid then size `x` must be >= `windowSize`
    assert(
      {
        let inputShape = x.shape &+ padding
        for i in 0..<S.rank {
          if windowSize[i] > inputShape[i] { return false }
        }
        return true
      }(), "input `x` plus `padding` must be >= the windowSize")

    pooling = PoolingDescriptor(
      mode: mode,
      nan: .noPropagate,
      windowSize: windowSize,
      padding: padding,
      strides: strides)

    // create input descriptor indenting dimensions with 1 as needed
    xDesc = TensorDescriptor(x, isBatch)

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
    var s = S.zero
    let base = xDesc.rank > S.rank ? xDesc.rank - S.rank : 0
    for i in 0..<S.rank {
      s[i] = Int(shape32[base + i])
    }
    outShape = s

    // create output descriptor
    outOrder = x.order
    outDesc = TensorDescriptor(Tensor<S, E>(shape: outShape, order: outOrder), isBatch)
  }

  @inlinable public func createOutput() -> Tensor<S, E> {
    Tensor<S, E>(shape: outShape, order: outOrder)
  }
}

//==============================================================================
/// PoolingDescriptor
public final class PoolingDescriptor<S: TensorShape> {
  // properties
  public let desc: cudnnPoolingDescriptor_t

  // initializers
  @inlinable public init(
    mode: PoolingMode,
    nan: NanPropagation,
    windowSize: S,
    padding: S,
    strides: S
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
        Int32(S.rank),
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
