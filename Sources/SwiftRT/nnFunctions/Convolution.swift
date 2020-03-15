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
public struct Convolution<T>: Layer
    where T: DifferentiableTensorView, T.Element: ScalarElement & Real
{
    /// The convolution filter
    public var filter: T
    /// The bias vector
    public var bias: T
    /// The element-wise activation function type
    @noDerivative public let activation: ActivationType
    /// The strides of the sliding window for spatial dimensions.
    @noDerivative public let strides: T.Bounds
    /// The padding algorithm for convolution.
    @noDerivative public let padding: Padding
    /// The dilation factor for spatial dimensions.
    @noDerivative public let dilations: T.Bounds
    /// The bounds of the output tensor
    @noDerivative public let outputBounds: T.Bounds
    /// device specific convolution operator
    @noDerivative public let deviceOp: DeviceConvolution<T>

    //--------------------------------------------------------------------------
    /// init
    /// creates and encapsulates a device specific convolution implementation
    @inlinable
    public init(
        for x: T,
        filter: T,
        bias: T,
        activation: ActivationType = .identity,
        strides: T.Bounds.Tuple = T.Bounds.oneTuple,
        padding: Padding = .valid,
        dilations: T.Bounds.Tuple = T.Bounds.oneTuple,
        properties: ConvolutionProperties = ConvolutionProperties())
    {
        self.filter = filter
        self.bias = bias
        self.activation = activation
        self.strides = T.Bounds(strides)
        self.padding = padding
        self.dilations = T.Bounds(dilations)
        
        do {
            // create the device op and save the output bounds
            var temp = T.Bounds.zero
            self.deviceOp =
                try Context.platform.currentQueue.convolution(
                    for: x, yBounds: &temp,
                    filter: filter, bias: bias,
                    activation: activation, strides: self.strides,
                    padding: padding, dilations: self.dilations,
                    properties: properties,
                    device: Context.platform.currentDevice,
                    filterBiasBackpropQueueIndex: 2)

            self.outputBounds = temp
            
        } catch {
            Context.platform.writeLog("\(error)")
            fatalError()
        }
    }

//    //--------------------------------------------------------------------------
//    /// - Parameters:
//    ///   - filterShape: The 3-D shape of the filter, representing
//    ///     (filter width, input channel count, output channel count).
//    ///   - stride: The stride of the sliding window for the temporal dimension.
//    ///   - padding: The padding algorithm for convolution.
//    ///   - dilation: The dilation factor for the temporal dimension.
//    ///   - activation: The element-wise activation function.
//    ///   - filterInitializer: Initializer to use for the filter parameters.
//    ///   - biasInitializer: Initializer to use for the bias parameters.
//    init(
//        filterShape: (Int, Int, Int),
//        stride: Int = 1,
//        padding: Padding = .valid,
//        dilation: Int = 1,
//        activation: ActivationType = .identity,
//        filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
//        biasInitializer: ParameterInitializer<Scalar> = zeros()
//    ) {
//
//    }
    
    //--------------------------------------------------------------------------
    @inlinable
    @differentiable
    public func callAsFunction(_ input: T) -> T {
        input
//        var y = T(bounds: outputBounds)
//        infer(y: &y, from: input, filter: filter, bias: bias)
//        return y
    }

//    //--------------------------------------------------------------------------
//    @inlinable
//    @differentiable
//    public func infer(y: inout T, from x: T, filter: T, bias: T) {
//        fatalError()
//    }
//
//    @inlinable
//    @derivative(of: infer)
//    public func _vjpInfer(y: inout T, from x: T, filter: T, bias: T)
//        -> (value: Void, pullback: (inout T) -> (T, T, T))
//    {
//        // Note: you may want to defer the original computation until
//        // returning the pullback to avoid mutating the value of `y`.
//        defer { infer(y: &y, from: x, filter: filter, bias: bias) }
//        return ((), { dy in
//            // TODO: Implement.
//            // Returns `(dx:dfilter:dbias:)`.
//            fatalError()
//        })
//    }
}

//==============================================================================
/// DeviceConvolution
/// an abstract base class used to manage device dependent
/// convolution implementations
public class DeviceConvolution<T>
    where T: DifferentiableTensorView, T.Element: ScalarElement
{
    public init() {}
    
    /// init
    /// initializes the device function `y = convolution(x)`
    /// - Parameter x: the input tensor
    /// - Parameter yBounds: the bounds of the output tensor `y`
    /// - Parameter filter: the convolution filter
    /// - Parameter bias: the filter bias
    /// - Parameter activation: the activation to be applied to the result
    /// - Parameter strides: the filter window strides
    /// - Parameter padding: the padding surrounding `x`
    /// - Parameter dilations: the dilations for the filter
    /// - Parameter properties: convolution customization properties
    /// - Parameter device: the device where the convolution will execute
    /// - Parameter filterBiasBackpropQueueIndex: the queue to use for filter
    /// and bias backpropagation
    public init(for x: T,
                yBounds: inout T.Bounds,
                filter: T,
                bias: T,
                activation: ActivationType,
                strides: T.Bounds,
                padding: Padding,
                dilations: T.Bounds,
                properties: ConvolutionProperties,
                device: ServiceDevice,
                filterBiasBackpropQueueIndex: Int) throws
    {
        fatalError("not implemented")
    }

    /// init
    /// initializes the device function `y = convolution(x)`
    /// - Parameter x: the input tensor
    /// - Parameter yBounds: the bounds of the output tensor `y`
    /// - Parameter filterBounds: the bounds of the filter to create
    /// - Parameter activation: the activation to be applied to the result
    /// - Parameter strides: the filter window strides
    /// - Parameter padding: the padding surrounding `x`
    /// - Parameter dilations: the dilations for the filter
    /// - Parameter properties: convolution customization properties
    /// - Parameter device: the device where the convolution will execute
    /// - Parameter filterBiasBackpropQueueIndex: the queue to use for filter
    /// and bias backpropagation
    public init(for x: T,
                yBounds: inout T.Bounds,
                filterBounds: T.Bounds,
                activation: ActivationType,
                strides: T.Bounds,
                padding: Padding,
                dilations: T.Bounds,
                properties: ConvolutionProperties,
                device: ServiceDevice,
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
    public func infer(y: inout T, from x: T, with filter: T, and bias: T)
        throws {
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
    public func backPropagate(y: T, yDiff: T,
                              filter: T, filterDiff: inout T,
                              bias: T, biasDiff: inout T,
                              x: T, xDiff: inout T) throws
    {
        fatalError("not implemented")
    }
}

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

