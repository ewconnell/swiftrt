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

public class CudaDense<T> where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // properties
    private let activation: ActivationMode
    private let activationDescriptor: ActivationDescriptor!
    private let biasTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor
    private var activationDiff: T!
    private let weight: T
    private let bias: T

    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout DataShape,
                weight: T,
                bias: T,
                activation: ActivationMode,
                nan: NanPropagation,
                reluCeiling: Double = 0) throws
    {
        assert(x.rank == 2 && weight.rank == 2 && bias.rank == 1)
        self.weight = weight
        self.bias = bias
        
        // activation
        self.activation = activation
        activationDescriptor = activation == .identity ? nil :
                try ActivationDescriptor(mode: activation,
                                         nan: nan,
                                         reluCeiling: reluCeiling)
        // setup bias
        biasTensorDescriptor = try bias.createTensorDescriptor()
        
        // return the shape of the output y and create a tensorDescriptor
        // with the same scalarType for y
        yShape = DataShape(extents: [x.extents[0], weight.extents[1]])
        yTensorDescriptor = try x.createTensorDescriptor(asShape: yShape)
        
        // allocate the `activationDiff` temporary tensor if the activation
        // is not identity and we are doing training
        if activation != .identity && DeviceContext.isTraining {
            activationDiff = x.createDense(with: yShape.extents)
        }
    }

    //--------------------------------------------------------------------------
    // inferring
    public func inferring(y: inout T, from x: T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        
        // TODO: is there a better fused kernel for Y = WX + b??
        // a row major matmul!
        try deviceQueue.gemm(
            transA: .noTranspose, matrixA: x,
            transB: .noTranspose, matrixB: weight,
            matrixC: &y)
        
        try cudaCheck(status: cudnnAddTensor(
            deviceQueue.cudnn.handle,
            // alpha
            T.Element.onePointer,
            // bias
            biasTensorDescriptor.desc,
            bias.deviceReadOnly(using: deviceQueue),
            // beta
            T.Element.zeroPointer,
            // y
            yTensorDescriptor.desc,
            y.deviceReadWrite(using: deviceQueue)))
        
        // activation TODO: verify that doing inplace is okay
        if activation != .identity {
            try cudaCheck(status: cudnnActivationForward(
                deviceQueue.cudnn.handle,
                activationDescriptor!.desc,
                // alpha
                T.Element.onePointer,
                // x
                yTensorDescriptor.desc,
                x.deviceReadOnly(using: deviceQueue),
                // beta
                T.Element.zeroPointer,
                // y
                yTensorDescriptor.desc,
                y.deviceReadWrite(using: deviceQueue)))
        }
    }

    //--------------------------------------------------------------------------
    // gradient
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws {
        let deviceQueue = DeviceContext.currentQueue as! CudaQueue
        
        if activation != .identity {
            try cudaCheck(status: cudnnActivationBackward(
                deviceQueue.cudnn.handle,
                activationDescriptor!.desc,
                // alpha
                T.Element.onePointer,
                // y
                yTensorDescriptor.desc,
                y.deviceReadOnly(using: deviceQueue),
                // dy
                yTensorDescriptor.desc,
                yDiff.deviceReadOnly(using: deviceQueue),
                // x
                yTensorDescriptor.desc,
                x.deviceReadOnly(using: deviceQueue),
                // beta
                T.Element.zeroPointer,
                // dx
                yTensorDescriptor.desc,
                activationDiff.deviceReadWrite(using: deviceQueue)))

            try deviceQueue.gemm(transA: .noTranspose, matrixA: activationDiff,
                                 transB: .transpose, matrixB: weight,
                                 matrixC: &xDiff)
        } else {
            try deviceQueue.gemm(transA: .noTranspose, matrixA: yDiff,
                                 transB: .transpose, matrixB: weight,
                                 matrixC: &xDiff)
        }
    }
}
