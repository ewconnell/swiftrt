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
// ConvolutionProperties
public struct ConvolutionProperties: Codable {
    public var activationNan: NanPropagation
    public var activationReluCeiling: Double
    public var backwardDataAlgorithm: ConvolutionBwdDataAlgorithm
    public var backwardDataWorkspaceLimit: Int
    public var backwardFilterAlgorithm: ConvolutionBwdFilterAlgorithm
    public var backwardFilterWorkspaceLimit: Int
    public var forwardAlgorithm: ConvolutionFwdAlgorithm
    public var forwardWorkspaceLimit: Int
    public var mode: ConvolutionMode
    
    @inlinable public init() {
        activationNan = .noPropagate
        activationReluCeiling = 0
        backwardDataAlgorithm = .fastest
        backwardDataWorkspaceLimit = 10.MB
        backwardFilterAlgorithm = .fastest
        backwardFilterWorkspaceLimit = 10.MB
        forwardAlgorithm = .fastest
        forwardWorkspaceLimit = 10.MB
        mode = .crossCorrelation
    }
}

//==============================================================================
// ConvolutionFwdAlgorithm
public enum ConvolutionFwdAlgorithm: Int, Codable, CaseIterable {
    case implicitGEMM
    case implicitPrecompGEMM
    case gemm
    case direct
    case fft
    case fftTiling
    case winograd
    case winogradNonFused
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionBwdDataAlgorithm
public enum ConvolutionBwdDataAlgorithm: Int, Codable, CaseIterable {
    case algo0
    case algo1
    case fft
    case fftTiling
    case winograd
    case winogradNonFused
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionBwdFilterAlgorithm
public enum ConvolutionBwdFilterAlgorithm: Int, Codable, CaseIterable {
    case algo0
    case algo1
    case algo3
    case fft
    case winograd
    case winogradNonFused
    case numAlgorithms
    case deterministic
    case fastest
    case noWorkspace
    case workspaceLimit
}

//==============================================================================
// ConvolutionMode
public enum ConvolutionMode: Int, Codable, CaseIterable {
    case convolution
    case crossCorrelation
}

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions
{
    public func convolution<Shape, Element, FilterElement>(
        activation: ActivationType,
        strides: Shape,
        padding: Padding,
        dilations: Shape,
        properties: ConvolutionProperties,
        deviceId: Int,
        filterBiasBackpropQueueIndex: Int
    ) -> DeviceConvolution<Shape, Element, FilterElement>
    where Shape: TensorShape,
          Element: ScalarElement,
          FilterElement: ScalarElement
    {
        CpuConvolution<Shape, Element, FilterElement>(
            activation: activation,
            strides: strides,
            padding: padding,
            dilations: dilations,
            properties: properties,
            deviceId: deviceId,
            filterBiasBackpropQueueIndex: filterBiasBackpropQueueIndex)
    }
}

//==============================================================================
/// CpuConvolution
public final class CpuConvolution<Shape, Element, FilterElement>:
    DeviceConvolution<Shape, Element, FilterElement>
where Shape: TensorShape,
      Element: ScalarElement,
      FilterElement: ScalarElement {}

//==============================================================================
/// DeviceConvolution
/// A convolution base class with a cpu default implementation. It is intended
/// to be overriden for Cuda and other accelerated implementations.
/// The input/output tensor type is of the form NWC, NHWC, or NDHWC
/// Filter dimensions are defined as follows:
/// [filter width, input channels, output channels]
/// [filter width, filter width, input channels, output channels]
/// [filter depth, filter width, filter width, input channels, output channels]
public class DeviceConvolution<Shape, Element, FilterElement>: Logging
where Shape: TensorShape,
      Element: ScalarElement,
      FilterElement: ScalarElement
{
    // types
    public typealias Data = Tensor<Shape, Element>
    public typealias Filter = Tensor<Shape, FilterElement>
    public typealias Bias = TensorR1<FilterElement>
    
    // properties
    public let properties: ConvolutionProperties
    public let padding: Padding
    public let strides: Shape
    public let dilations: Shape
    public var inputShape: Shape

    //--------------------------------------------------------------------------
    /// init
    /// - Parameters:
    ///  - activation: the activation to be applied to the result
    ///  - strides: the filter window strides
    ///  - padding: the padding surrounding `x`
    ///  - dilations: the dilations for the filter
    ///  - properties: convolution customization properties
    ///  - device: the device where the convolution will execute
    ///  - filterBiasBackpropQueueIndex: the queue to use for filter
    /// and bias backpropagation
    @inlinable public init(
        activation: ActivationType,
        strides: Shape,
        padding: Padding,
        dilations: Shape,
        properties: ConvolutionProperties,
        deviceId: Int,
        filterBiasBackpropQueueIndex: Int
    ) {
        //----------------------------------
        // save properties
        self.properties = properties
        self.padding = padding
        self.strides = strides
        self.dilations = dilations
        self.inputShape = Shape.zero
    }
    
    //--------------------------------------------------------------------------
    /// forward
    /// - Parameter y: the output tensor
    /// - Parameter x: the input tensor
    /// - Parameter filter: the convolution filter
    /// - Parameter bias: the filter bias
    //    @differentiable
    @inlinable public func forward(
        x: Data,
        filter: Filter,
        bias: Bias
    ) -> Data {
        fatalError("abstract not implemented")
    }

    //--------------------------------------------------------------------------
    /// backward
    /// - Parameter y: the output tensor
    /// - Parameter yDiff: the output differential
    /// - Parameter filter: the convolution filter
    /// - Parameter filterDiff: the filter differential
    /// - Parameter bias: the filter bias
    /// - Parameter biasDiff: the filter bias differential
    /// - Parameter x: the input tensor
    /// - Parameter x: the input tensor differential
    @inlinable public func backward(
        y: Data,
        yDiff: Data,
        filter: Filter,
        filterDiff: inout Filter,
        bias: Bias,
        biasDiff: inout Bias,
        x: Data,
        xDiff: inout Data
    ) {
        fatalError("abstract not implemented")
    }
}
