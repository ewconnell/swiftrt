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
import Foundation
import Numerics

//==============================================================================
/// Convolution
public struct Convolution<T, F> where
    T: DifferentiableTensorView, T.Element: ScalarElement,
    F: TensorView, F.Bounds == T.Bounds,
    F.Element: ScalarElement & BinaryFloatingPoint
{
    public typealias BiasVector = Vector<F.Element>
    /// The convolution filter
    public var filter: F
    /// The bias vector
    public var bias: BiasVector?
    /// The element-wise activation function type
    @noDerivative public let activation: ActivationType
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: T.Bounds
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: T.Bounds
    /// device specific convolution operator
    @noDerivative public let deviceOp: CudaConvolution<T, F>

    //--------------------------------------------------------------------------
    /// Creates a `Convolution` layer with the specified filter, bias,
    /// activation function, stride, dilation and padding.
    ///
    /// - Parameters:
    ///   - filter: The convolution filter of shape
    ///     [filter width, input channel count, output channel count].
    ///   - bias: The bias vector of shape [output channel count].
    ///   - activation: The element-wise activation function.
    ///   - strides: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    @inlinable
    public init(
        filter: F,
        bias: BiasVector? = nil,
        activation: ActivationType = .identity,
        strides: T.Bounds = T.Bounds.one,
        padding: Padding = .valid,
        dilations: T.Bounds = T.Bounds.one,
        properties: ConvolutionProperties = ConvolutionProperties())
    {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        
        do {
            // create the device op and save the output bounds
            self.deviceOp =
                try Context.platform.currentQueue.convolution(
                    activation: activation,
                    strides: self.strides,
                    padding: padding,
                    dilations: self.dilations,
                    properties: properties,
                    device: Context.platform.currentDevice,
                    filterBiasBackpropQueueIndex: 2)
        } catch {
            Context.platform.writeLog("\(error)")
            fatalError()
        }
    }
    
    //--------------------------------------------------------------------------
    /// Creates a `Convolution` layer with the specified filter shape,
    /// stride, padding, dilation and element-wise activation function.
    ///
    /// - Parameters:
    ///   - filterShape: The 3-D shape of the filter, representing
    ///     (filter width, input channel count, output channel count).
    ///   - stride: The stride of the sliding window for the temporal dimension.
    ///   - padding: The padding algorithm for convolution.
    ///   - dilation: The dilation factor for the temporal dimension.
    ///   - activation: The element-wise activation function.
    ///   - filterInitializer: Initializer to use for the filter parameters.
    ///   - biasInitializer: Initializer to use for the bias parameters.
    @inlinable
    init(
        filterShape: F.Bounds,
        stride: Int = 1,
        padding: Padding = .valid,
        dilation: Int = 1,
        activation: ActivationType = .identity,
        filterInitializer: ParameterInitializer<F>,
        biasInitializer: ParameterInitializer<BiasVector>? = nil
    ) {
        let biasBounds = Bounds1(filterShape[F.rank - 1])
        let bias: BiasVector? = biasInitializer == nil ? nil :
            biasInitializer!(biasBounds)
        
        self.init(filter: filterInitializer(filterShape),
                  bias: bias,
                  activation: activation,
                  strides: T.Bounds(repeating: stride),
                  padding: padding,
                  dilations: T.Bounds(repeating: dilation))
    }
}

////==============================================================================
///// DeviceConvolution
///// an abstract base class used to manage device dependent
///// convolution implementations
///// - Parameter T: the input/output tensor type, which is expected to be
///// of the form NWC, NHWC, or NDHWC
///// Filter dimensions are defined as follows:
///// [filter width, input channels, output channels]
///// [filter width, filter width, input channels, output channels]
///// [filter depth, filter width, filter width, input channels, output channels]
//public class DeviceConvolution<T>
//    where T: DifferentiableTensorView, T.Element: ScalarElement
//{
//    public init() {}
//
//    /// init
//    /// initializes the device function `y = convolution(x)`
//    /// - Parameter filter: the convolution filter
//    /// - Parameter bias: the filter bias vector
//    /// - Parameter activation: the activation to be applied to the result
//    /// - Parameter strides: the filter window strides
//    /// - Parameter padding: the padding surrounding `x`
//    /// - Parameter dilations: the dilations for the filter
//    /// - Parameter properties: convolution customization properties
//    /// - Parameter device: the device where the convolution will execute
//    /// - Parameter filterBiasBackpropQueueIndex: the queue to use for filter
//    /// and bias backpropagation
//    public init(activation: ActivationType,
//                strides: T.Bounds,
//                padding: Padding,
//                dilations: T.Bounds,
//                properties: ConvolutionProperties,
//                device: ServiceDevice,
//                filterBiasBackpropQueueIndex: Int) throws
//    {
//        fatalError("not implemented")
//    }
//
//    /// infer
//    /// - Parameter y: the output tensor
//    /// - Parameter x: the input tensor
//    /// - Parameter filter: the convolution filter
//    /// - Parameter bias: the filter bias
////    @differentiable
//    public func infer<F>(from x: T, with filter: F, and bias: Vector<F.Element>)
//        throws -> T
//        where F: TensorView, F.Bounds == T.Bounds, F.Element == T.Element
//    {
//        fatalError("not implemented")
//    }
//
//    /// backPropagate
//    /// - Parameter y: the output tensor
//    /// - Parameter yDiff: the output differential
//    /// - Parameter filter: the convolution filter
//    /// - Parameter filterDiff: the filter differential
//    /// - Parameter bias: the filter bias
//    /// - Parameter biasDiff: the filter bias differential
//    /// - Parameter x: the input tensor
//    /// - Parameter x: the input tensor differential
//    public func backPropagate<F>(y: T, yDiff: T,
//                                 filter: F,
//                                 filterDiff: inout F,
//                                 bias: Vector<F.Element>,
//                                 biasDiff: inout Vector<F.Element>,
//                                 x: T, xDiff: inout T) throws
//        where F: TensorView, F.Bounds == T.Bounds, F.Element == T.Element
//    {
//        fatalError("not implemented")
//    }
//}

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
    
    @inlinable
    public init() {
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

