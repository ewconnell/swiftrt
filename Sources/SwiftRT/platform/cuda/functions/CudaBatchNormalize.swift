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

public struct CudaBatchNormalize<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private let deviceQueue: CudaQueue
    private let epsilon: Double
    private let expAverageFactor: Double
    private let normalizeMode: cudnnBatchNormMode_t
    private let xyTensorDescriptor: TensorDescriptor
    private let scaleBiasMeanVarianceTensorDescriptor: TensorDescriptor

    // working buffers
    private var workspaceSize = 0
    private var saved_mean: DeviceArray!
    private var saved_var: DeviceArray!
    private var grad_scale: DeviceArray!
    private var grad_bias: DeviceArray!
    private var scale: T
    private var bias: DeviceArray

    
    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout DataShape,
                epsilon: Double,
                momentum: Double,
                mode: BatchNormalizeMode) throws
    {
        assert(epsilon >= CUDNN_BN_MIN_EPSILON, "epsilon must be greater " +
            "than or equal to: \(CUDNN_BN_MIN_EPSILON)")
        self.deviceQueue = DeviceContext.currentQueue as! CudaQueue
        self.epsilon = epsilon
        self.normalizeMode = mode.cudnn
        self.expAverageFactor = 1.0 - momentum
        
        // xy
        xyTensorDescriptor = try x.createTensorDescriptor()
        yShape = x.shape

        // create the scaleBiasMeanVarianceTensor descriptor
        var temp: cudnnTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
        try cudaCheck(status: cudnnDeriveBNTensorDescriptor(
            temp!, xyTensorDescriptor.desc, normalizeMode))
        scaleBiasMeanVarianceTensorDescriptor = TensorDescriptor(owning: temp!)

        //--------------------------------
        // create workspace device arrays
        let (extents, strides, _) =
            try scaleBiasMeanVarianceTensorDescriptor.getInfo()
        let shape = DataShape(extents: extents, strides: strides)
        workspaceSize = shape.elementCount * MemoryLayout<T.Element>.size
        
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        var ones = x.createDense(with: shape.extents)
        deviceQueue.fill(result: &ones, with: 1)
        scale = ones
        
        func createZeroWorkspace(byteCount: Int) throws -> DeviceArray {
            return try deviceQueue.device.createArray(byteCount: byteCount,
                                                      heapIndex: 0,
                                                      zero: true)
        }
        
        bias = try createZeroWorkspace(byteCount: workspaceSize)
        
        if DeviceContext.isTraining {
            saved_mean = try createZeroWorkspace(byteCount: workspaceSize)
            saved_var = try createZeroWorkspace(byteCount: workspaceSize)
        }
    }
//
//    //----------------------------------------------------------------------------
//    // createZeroWorkspace
//    private func createZeroWorkspace(byteCount: Int) throws -> DeviceArray {
//        let array = try deviceQueue.device.createArray(byteCount: byteCount,
//                                                       heapIndex: 0)
//        try deviceQueue.zero(array: array)
//        return array
//    }
    
    //--------------------------------------------------------------------------
    // inferring
    //
    public func inferring(y: inout T, from x: T) throws {
        
//        if DeviceContext.current.isInferring {
//            try cudaCheck(status: cudnnBatchNormalizationForwardInference(
//                deviceQueue.cudnn.handle,
//                normalizeMode,
//                // alpha
//                T.Element.onePointer,
//                // beta
//                T.Element.zeroPointer,
//                // x
//                xyTensorDescriptor.desc,
//                inData.ro(using: dataStream),
//                // y
//                xyTensorDescriptor.desc,
//                outData.rw(using: dataStream),
//                //
//                scaleBiasMeanVarianceTensor!.desc,
//                scale.ro(using: dataStream),
//                bias.data,
//                props.running_mean?.ro(using: dataStream),
//                props.running_var?.ro(using: dataStream),
//                props.epsilon
//            ))
//        } else {
//            try cudaCheck(status: cudnnBatchNormalizationForwardTraining(
//                deviceQueue.cudnn.handle,
//                normalizeMode,
//                inData.one,
//                outData.zero,
//                inTensor.desc,
//                inData.ro(using: dataStream),
//                outTensor.desc,
//                outData.rw(using: dataStream),
//                scaleBiasMeanVarianceTensor!.desc,
//                scale.ro(using: dataStream),
//                bias.data,
//                expAverageFactor,
//                props.running_mean?.rw(using: dataStream),
//                props.running_var?.rw(using: dataStream),
//                props.epsilon,
//                saved_mean.data,
//                saved_var.data
//            ))
//        }

    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
//        let deviceQueue = DeviceContext.currentQueue as! CudaQueue

    }
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension BatchNormalizeMode {
    public var cudnn: cudnnBatchNormMode_t {
        get {
            switch self {
            case .perActivation: return CUDNN_BATCHNORM_PER_ACTIVATION
            case .spatial: return CUDNN_BATCHNORM_SPATIAL
            }
        }
    }
}
