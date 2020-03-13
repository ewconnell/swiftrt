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

////==============================================================================
///// Convolution
//public struct Convolution<T>: Layer
//    where
//    T: DifferentiableTensorView, T.Element: Real,
//    T: RealFunctions & ElementaryFunctions
//{
//    /// The convolution filter
//    public var filter: T
//    /// The bias vector
//    public var bias: T
//    /// The element-wise activation function type
//    @noDerivative public let activation: ActivationType
//    /// The strides of the sliding window for spatial dimensions.
//    @noDerivative public let strides: T.Bounds
//    /// The padding algorithm for convolution.
//    @noDerivative public let padding: Padding
//    /// The dilation factor for spatial dimensions.
//    @noDerivative public let dilation: T.Bounds
//
//    public init(
//        for tensor: T,
//        resultShape: inout Shape<T.Bounds>,
//        filter: T,
//        bias: T,
//        activation: ActivationType = .identity,
//        strides: T.Bounds = T.Bounds.one,
//        padding: Padding = .valid,
//        dilation: T.Bounds = T.Bounds.one)
//    {
//        self.filter = filter
//        self.bias = bias
//        self.activation = activation
//        self.strides = strides
//        self.padding = padding
//        self.dilation = dilation
//    }
//
//    public func callAsFunction(_ input: T) -> T {
//        fatalError()
//    }
//}
//
//==============================================================================
// ConvolutionProperties
public struct ConvolutionProperties: Codable {
    var activationNan: NanPropagation = .noPropagate
    var activationReluCeiling: Double = 0
    var backwardDataAlgorithm: ConvolutionBwdDataAlgorithm = .fastest
    var backwardDataWorkspaceLimit: Int = 10.MB
    var backwardFilterAlgorithm: ConvolutionBwdFilterAlgorithm = .fastest
    var backwardFilterWorkspaceLimit: Int = 10.MB
    var forwardAlgorithm: ConvolutionFwdAlgorithm = .fastest
    var forwardWorkspaceLimit: Int = 10.MB
    var mode: ConvolutionMode = .crossCorrelation
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

