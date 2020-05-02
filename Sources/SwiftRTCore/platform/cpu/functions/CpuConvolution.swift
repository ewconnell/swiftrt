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

//==============================================================================
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions & CpuMapOps
{
    public func convolution<Shape,E,FE>(
        activation: ActivationType,
        strides: Shape,
        padding: Padding,
        dilations: Shape,
        properties: ConvolutionProperties,
        deviceId: Int,
        filterBiasBackpropQueueIndex: Int
    ) -> DeviceConvolution<Shape,E,FE>
    where Shape: TensorShape,
          E: ScalarElement, FE: ScalarElement & BinaryFloatingPoint
    {
        cpu_convolution(
            activation: activation, strides: strides,
            padding: padding, dilations: dilations,
            properties: properties, deviceId: deviceId,
            filterBiasBackpropQueueIndex: filterBiasBackpropQueueIndex)
    }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions {
    public func cpu_convolution<Shape,E,FE>(
        activation: ActivationType,
        strides: Shape,
        padding: Padding,
        dilations: Shape,
        properties: ConvolutionProperties,
        deviceId: Int,
        filterBiasBackpropQueueIndex: Int
    ) -> DeviceConvolution<Shape,E,FE>
    where Shape: TensorShape,
          E: ScalarElement, FE: ScalarElement & BinaryFloatingPoint
    {
        fatalError("cpu convolution not implemented")
    }
}
