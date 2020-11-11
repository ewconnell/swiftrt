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

import SwiftRTCuda

//==============================================================================
/// CudaActivation
public final class CudaActivation<Shape, Element>: Logging
where Shape: TensorShape, Element: StorageElement & FloatingPoint {
  // types
  public typealias Input = Tensor<Shape, Element>
  public typealias Output = Tensor<Shape, Element>

  // constants
  public var zero = Element.zero
  public var one = Element.one

  // properties
  public let activationDescriptor: ActivationDescriptor
  public var xyTensorDescriptor: TensorDescriptor!
  public let deviceQueue: CudaQueue
  public var inputShape: Shape

  // retained tensors
  public var y: Output!

  //--------------------------------------------------------------------------
  // initializer

  public init(
    x: Input,
    mode: ActivationType,
    nan: NanPropagation,
    reluCeiling: Double = 0
  ) {
    deviceQueue = currentQueue
    inputShape = Shape.zero

    activationDescriptor = ActivationDescriptor(
      mode: mode,
      nan: nan,
      reluCeiling: reluCeiling)
  }

  //--------------------------------------------------------------------------
  // forward
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationForward
  @inlinable public func forward(x: Input) -> Output {
    // setup any time the input shape changes
    if x.shape != inputShape {
      setupForward(x)
    }

    cudaCheck(
      cudnnActivationForward(
        deviceQueue.cudnn.handle,
        activationDescriptor.desc,
        // alpha
        &one,
        // x
        xyTensorDescriptor.desc,
        x.deviceRead(using: deviceQueue),
        // beta
        &zero,
        // y
        xyTensorDescriptor.desc,
        y.deviceReadWrite(using: deviceQueue)))

    return y
  }

  //--------------------------------------------------------------------------
  // setupForward
  @inlinable public func setupForward(_ x: Input) {
    // TODO: figure out how S4TF wants to handle layouts
    // create tensor descriptors
    //        let tensorShape = inData.layout != .matrix ? inData.shape :
    //            Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)

    // TODO: set isBatch to true for now
    xyTensorDescriptor = TensorDescriptor(x)
    y = Tensor(like: x)
  }

  //--------------------------------------------------------------------------
  // backward
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward
  @inlinable public func backward(
    y: Output,
    yDiff: Output,
    x: Input,
    xDiff: inout Input
  ) {
    cudaCheck(
      cudnnActivationBackward(
        deviceQueue.cudnn.handle,
        activationDescriptor.desc,
        // alpha
        &one,
        // y
        xyTensorDescriptor.desc,
        y.deviceRead(using: deviceQueue),
        // dy
        xyTensorDescriptor.desc,
        yDiff.deviceRead(using: deviceQueue),
        // x
        xyTensorDescriptor.desc,
        x.deviceRead(using: deviceQueue),
        // beta
        &zero,
        // dx
        xyTensorDescriptor.desc,
        xDiff.deviceReadWrite(using: deviceQueue)))
  }
}

//==============================================================================
// ActivationDescriptor
public final class ActivationDescriptor {
  // properties
  public let desc: cudnnActivationDescriptor_t

  // initializers
  @inlinable public init(
    mode: ActivationType,
    nan: NanPropagation,
    reluCeiling: Double
  ) {
    // create the descriptor
    var temp: cudnnActivationDescriptor_t?
    cudaCheck(cudnnCreateActivationDescriptor(&temp))
    desc = temp!

    // initialize
    cudaCheck(
      cudnnSetActivationDescriptor(
        desc, mode.cudnn, nan.cudnn, reluCeiling))
  }

  @inlinable deinit {
    cudaCheck(cudnnDestroyActivationDescriptor(desc))
  }
}

//==============================================================================
extension ActivationType {
  public var cudnn: cudnnActivationMode_t {
    let modes: [ActivationType: cudnnActivationMode_t] = [
      .sigmoid: CUDNN_ACTIVATION_SIGMOID,
      .relu: CUDNN_ACTIVATION_RELU,
      .tanh: CUDNN_ACTIVATION_TANH,
      .clippedRelu: CUDNN_ACTIVATION_CLIPPED_RELU,
      .elu: CUDNN_ACTIVATION_ELU,
      .identity: CUDNN_ACTIVATION_IDENTITY,
    ]
    return modes[self]!
  }
}
