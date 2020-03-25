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

// *** TODO design questions!
// 1) class or struct, how are things retained, reused, etc?
// 2) should the input be retained to guarentee that init
//    matches the same shape as inferring? Or just assert in inferring?

public struct CudaSoftmax<T> where
    T: TensorView, T.Element: ScalarElement
{
    // properties
    private var cudnnAlgorithm: cudnnSoftmaxAlgorithm_t
    private var cudnnMode: cudnnSoftmaxMode_t
    private let tensorDescriptor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout Shape<T.Bounds>,
                algorithm: SoftmaxAlgorithm,
                mode: SoftmaxMode) throws
    {
        cudnnAlgorithm = algorithm.cudnn
        cudnnMode = mode.cudnn
        
        // create x and y tensor descriptor
        tensorDescriptor = try x.createTensorDescriptor()

        // return the shape of the output y
        yShape = x.shape
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxForward
    public func inferring(y: inout T, from x: T) throws {
        let deviceQueue = Context.service.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnSoftmaxForward(
            deviceQueue.cudnn.handle,
            cudnnAlgorithm,
            cudnnMode,
            // alpha
            T.Element.onePointer,
            // x
            tensorDescriptor.desc,
            x.deviceRead(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // y
            tensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))
    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxBackward
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
        let deviceQueue = Context.service.currentQueue as! CudaQueue

        // if there aren't any labels then do a normal backward
        try cudaCheck(status: cudnnSoftmaxBackward(
            deviceQueue.cudnn.handle,
            cudnnAlgorithm,
            cudnnMode,
            // alpha
            T.Element.onePointer,
            // y
            tensorDescriptor.desc,
            y.deviceRead(using: deviceQueue),
            // dy
            tensorDescriptor.desc,
            yDiff.deviceRead(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // dx
            tensorDescriptor.desc,
            xDiff.deviceReadWrite(using: deviceQueue)))
    }
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension SoftmaxAlgorithm {
    public var cudnn: cudnnSoftmaxAlgorithm_t {
        get {
            let algorithms: [SoftmaxAlgorithm: cudnnSoftmaxAlgorithm_t] = [
                .accurate: CUDNN_SOFTMAX_ACCURATE,
                .fast: CUDNN_SOFTMAX_FAST,
                .log: CUDNN_SOFTMAX_LOG,
            ]
            return algorithms[self]!
        }
    }
}

extension SoftmaxMode {
    public var cudnn: cudnnSoftmaxMode_t {
        get {
            switch self {
            case .channel : return CUDNN_SOFTMAX_MODE_CHANNEL
            case .instance: return CUDNN_SOFTMAX_MODE_INSTANCE
            }
        }
    }
}
