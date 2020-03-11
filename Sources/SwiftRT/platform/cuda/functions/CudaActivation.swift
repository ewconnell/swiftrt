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
import CCuda

//==============================================================================
/// CudaActivationInferring
/// used to do activation inference
open class CudaActivationInferring<T>: ActivationTraining<T>
    where T: TensorView, T.Element: ScalarElement & FloatingPoint
{
    // properties
    public let activationDescriptor: ActivationDescriptor
    public let xyTensorDescriptor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                y: inout T,
                mode: ActivationMode,
                nan: NanPropagation,
                reluCeiling: Double = 0) throws
    {
        // create descriptor
        activationDescriptor =
            try ActivationDescriptor(mode: mode, nan: nan,
                                     reluCeiling: reluCeiling)
        
        // TODO: figure out how S4TF wants to handle layouts
        // create tensor descriptors
        //        let tensorShape = inData.layout != .matrix ? inData.shape :
        //            Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)
        
        xyTensorDescriptor = try x.createTensorDescriptor()

        // return correctly sized storage for `y`
        // in this case x and y are the same size
        y = x.createDense()
    }

    //--------------------------------------------------------------------------
    // infer
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationForward
    public override func infer(y: inout T, from x: T) throws {
        let deviceQueue = Platform.service.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnActivationForward(
            deviceQueue.cudnn.handle,
            activationDescriptor.desc,
            // alpha
            T.Element.onePointer,
            // x
            xyTensorDescriptor.desc,
            x.deviceRead(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // y
            xyTensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))
    }
}

//==============================================================================
/// CudaActivationTraining
/// used to do activation inference and training
public final class CudaActivationTraining<T>: CudaActivationInferring<T>
    where T: TensorView, T.Element: ScalarElement & FloatingPoint
{
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward
    public override func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
        let deviceQueue = Platform.service.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnActivationBackward(
            deviceQueue.cudnn.handle,
            activationDescriptor.desc,
            // alpha
            T.Element.onePointer,
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
            T.Element.zeroPointer,
            // dx
            xyTensorDescriptor.desc,
            xDiff.deviceReadWrite(using: deviceQueue)))
    }
}

//==============================================================================
// ActivationDescriptor
public final class ActivationDescriptor : ObjectTracking {
    // properties
    public let trackingId: Int
    public let desc: cudnnActivationDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(mode: ActivationMode,
                nan: NanPropagation,
                reluCeiling: Double) throws
    {
        // create the descriptor
        var temp: cudnnActivationDescriptor_t?
        try cudaCheck(status: cudnnCreateActivationDescriptor(&temp))
        desc = temp!

        // initialize
        try cudaCheck(status: cudnnSetActivationDescriptor(
            desc, mode.cudnn, nan.cudnn, reluCeiling))
        trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    // deinit
    @inlinable
    deinit {
        try! cudaCheck(status: cudnnDestroyActivationDescriptor(desc))
        ObjectTracker.global.remove(trackingId: trackingId)
    }
}

//==============================================================================
extension ActivationMode {
    public var cudnn: cudnnActivationMode_t {
        get {
            let modes: [ActivationMode: cudnnActivationMode_t] = [
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
}

