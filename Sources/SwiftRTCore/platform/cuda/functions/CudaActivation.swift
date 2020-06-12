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
/// CudaActivation
public final class CudaActivation<Shape, Element>: Logging
where Shape: TensorShape, Element: ScalarElement & FloatingPoint
{
    // types
    public typealias Input = Tensor<Shape,Element>
    public typealias Output = Tensor<Shape,Element>

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
        deviceQueue = Context.currentQueue
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
        do {
            // setup any time the input shape changes
            if x.shape != inputShape {
                setupForward(x)
            }
            
            try cudaCheck(status: cudnnActivationForward(
                deviceQueue.cudnn.handle,
                activationDescriptor.desc,
                // alpha
                Element.onePointer,
                // x
                xyTensorDescriptor.desc,
                x.deviceRead(using: deviceQueue),
                // beta
                Element.zeroPointer,
                // y
                xyTensorDescriptor.desc,
                y.deviceReadWrite(using: deviceQueue)))
        } catch {
            writeLog("\(error)")
            fatalError()
        }
        return y
    }
    
    //--------------------------------------------------------------------------
    // setupForward
    @inlinable public func setupForward(_ x: Input) {
        // TODO: figure out how S4TF wants to handle layouts
        // create tensor descriptors
        //        let tensorShape = inData.layout != .matrix ? inData.shape :
        //            Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)
        
        xyTensorDescriptor = x.createTensorDescriptor()
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
        do {
            try cudaCheck(status: cudnnActivationBackward(
                deviceQueue.cudnn.handle,
                activationDescriptor.desc,
                // alpha
                Element.onePointer,
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
                Element.zeroPointer,
                // dx
                xyTensorDescriptor.desc,
                xDiff.deviceReadWrite(using: deviceQueue)))
        } catch {
            writeLog("\(error)")
            fatalError()
        }
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
        do {
            // create the descriptor
            var temp: cudnnActivationDescriptor_t?
            try cudaCheck(status: cudnnCreateActivationDescriptor(&temp))
            desc = temp!
            
            // initialize
            try cudaCheck(status: cudnnSetActivationDescriptor(
                            desc, mode.cudnn, nan.cudnn, reluCeiling))
        } catch {
            Context.currentQueue.writeLog(
                "\(createString) \(Self.self) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyActivationDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
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

