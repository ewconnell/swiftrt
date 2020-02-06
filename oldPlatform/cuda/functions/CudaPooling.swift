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

public struct CudaPooling<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private let poolingDescriptor: PoolingDescriptor
    private let xTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout DataShape,
                filterSize: [Int],
                strides: [Int],
                padding: [Int],
                poolingMode: PoolingMode,
                nan: NanPropagation) throws
    {
        // create the descriptor
        poolingDescriptor = try PoolingDescriptor(
            mode: poolingMode,
            nan: nan,
            filterSize: filterSize,
            padding: padding,
            strides: strides)

        // create input tensor descriptor
        xTensorDescriptor = try x.createTensorDescriptor()

        // get output extents based on settings
        var extents = [Int32](repeating: 0, count: x.rank)
        try cudaCheck(status: cudnnGetPoolingNdForwardOutputDim(
            poolingDescriptor.desc,
            xTensorDescriptor.desc,
            Int32(x.rank),
            &extents))

        // return the shape of the output y and create a tensorDescriptor
        // with the same scalarType for y as x
        yShape = DataShape(extents: extents.map { Int($0) })
        yTensorDescriptor = try x.createTensorDescriptor(asShape: yShape)
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingForward
    public func inferring(y: inout T, from x: T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnPoolingForward(
            deviceQueue.cudnn.handle,
            poolingDescriptor.desc,
            // alpha
            T.Element.onePointer,
            // x
            xTensorDescriptor.desc,
            x.deviceReadOnly(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // y
            yTensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))
    }
    
    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingBackward
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        
        try cudaCheck(status: cudnnPoolingBackward(
            deviceQueue.cudnn.handle,
            poolingDescriptor.desc,
            // alpha
            T.Element.onePointer,
            // y
            yTensorDescriptor.desc,
            y.deviceReadOnly(using: deviceQueue),
            // dy
            yTensorDescriptor.desc,
            yDiff.deviceReadOnly(using: deviceQueue),
            // x
            xTensorDescriptor.desc,
            x.deviceReadOnly(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // dx
            xTensorDescriptor.desc,
            xDiff.deviceReadWrite(using: deviceQueue)))
    }
}

//==============================================================================
// PoolingMode
extension cudnnPoolingMode_t : Hashable {}

extension PoolingMode {
    public var cudnn: cudnnPoolingMode_t {
        get {
            let modes: [PoolingMode: cudnnPoolingMode_t] = [
                .max: CUDNN_POOLING_MAX,
                .maxDeterministic: CUDNN_POOLING_MAX_DETERMINISTIC,
                .averageExcludePadding: CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                .averageIncludePadding: CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            ]
            return modes[self]!
        }
    }
}
