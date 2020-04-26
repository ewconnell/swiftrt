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

import Foundation

//==============================================================================
/// DeviceConvolution
/// an abstract base class used to manage device dependent
/// convolution implementations
/// - Parameter T: the input/output tensor type, which is expected to be
/// of the form NWC, NHWC, or NDHWC
/// Filter dimensions are defined as follows:
/// [filter width, input channels, output channels]
/// [filter width, filter width, input channels, output channels]
/// [filter depth, filter width, filter width, input channels, output channels]
public class DeviceConvolution<Shape,E,FE>
where Shape: TensorShape,
      E: ScalarElement, FE: ScalarElement & BinaryFloatingPoint
{
    public typealias Data = Tensor<Shape,E>
    public typealias Filter = Tensor<Shape,FE>
    public typealias Bias = Tensor1<FE>
    
    public init() {}
    
    /// init
    /// initializes the device function `y = convolution(x)`
    /// - Parameter filter: the convolution filter
    /// - Parameter bias: the filter bias vector
    /// - Parameter activation: the activation to be applied to the result
    /// - Parameter strides: the filter window strides
    /// - Parameter padding: the padding surrounding `x`
    /// - Parameter dilations: the dilations for the filter
    /// - Parameter properties: convolution customization properties
    /// - Parameter device: the device where the convolution will execute
    /// - Parameter filterBiasBackpropQueueIndex: the queue to use for filter
    /// and bias backpropagation
    public init(activation: ActivationType,
                strides: Shape,
                padding: Padding,
                dilations: Shape,
                properties: ConvolutionProperties,
                deviceId: Int,
                filterBiasBackpropQueueIndex: Int) throws
    {
        fatalError("not implemented")
    }
    
    /// infer
    /// - Parameter y: the output tensor
    /// - Parameter x: the input tensor
    /// - Parameter filter: the convolution filter
    /// - Parameter bias: the filter bias
    //    @differentiable
    public func infer(from x: Data, with filter: Filter, and bias: Bias)
    throws -> Data
    {
        fatalError("not implemented")
    }
    
    /// backPropagate
    /// - Parameter y: the output tensor
    /// - Parameter yDiff: the output differential
    /// - Parameter filter: the convolution filter
    /// - Parameter filterDiff: the filter differential
    /// - Parameter bias: the filter bias
    /// - Parameter biasDiff: the filter bias differential
    /// - Parameter x: the input tensor
    /// - Parameter x: the input tensor differential
    public func backPropagate(
        y: Data,
        yDiff: Data,
        filter: Filter,
        filterDiff: inout Filter,
        bias: Bias,
        biasDiff: inout Bias,
        x: Data,
        xDiff: inout Data
    ) throws
    {
        fatalError("not implemented")
    }
}
