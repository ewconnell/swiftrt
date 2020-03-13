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
import CCuda
import Numerics
//import CudaKernels

//==============================================================================
/// CudaQueue
public final class CudaQueue: DeviceQueue {
    public let useGpu: Bool
    
    public let stream: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                deviceId: Int, deviceName: String, useGpu: Bool) throws
    {
        self.useGpu = useGpu
        
        // select the specified device
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
        
        // create a queue associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
        stream = cudaStream!
        cudnn = try CudnnHandle(deviceId: deviceId, using: stream)
        cublas = try CublasHandle(deviceId: deviceId, using: stream)

        super.init(id: id, parent: logInfo, deviceId: deviceId,
                   deviceName: deviceName,
                   memoryType: useGpu ? .discreet : .unified)
    }
    
    //--------------------------------------------------------------------------
    // select
    public func selectDevice() throws {
        try cudaCheck(status: cudaSetDevice(Int32(self.deviceId)))
    }
    
    //==========================================================================
    /// createActivation
//    public override func createActivation<T>(
//        x: T,
//        y: inout T,
//        mode: ActivationType,
//        nan: NanPropagation,
//        reluCeiling: Double = 0) throws -> ActivationInferring<T>
//        where T: TensorView, T.Element: ScalarElement & FloatingPoint
//    {
//        return try CudaActivationInferring(x: x, y: &y, mode: mode,
//                                           nan: nan, reluCeiling: reluCeiling)
//    }

    //==========================================================================
    // convolution
    public override func convolution<T>(
        for x: T,
        yShape: inout Shape<T.Bounds>,
        filter: T,
        bias: T,
        activation: ActivationType,
        strides: T.Bounds,
        padding: Padding,
        dilations: T.Bounds,
        properties: ConvolutionProperties,
        device: ServiceDevice,
        filterBiasBackpropQueueIndex: Int) throws -> DeviceConvolution<T>
        where T: DifferentiableTensorView, T.Element: ScalarElement & Real
    {
        fatalError("cpu convolution not implemented")
    }
}



//public class ConvolutionInferring<T> where
//    T: TensorView, T.Element: FloatingPoint
//{
//    public func infer(y: inout T, from x: T, filter: T, bias: T) throws
//    { fatalError("Abstract") }
//}
//
//public final class ConvolutionTraining<T>: ConvolutionInferring<T> where
//    T: TensorView, T.Element: FloatingPoint
//{
//    public func gradient(y: T, yDiff: T,
//                         filter: T, filterDiff: inout T,
//                         bias: T, biasDiff: inout T,
//                         x: T, xDiff: inout T) throws
//    { fatalError("Abstract") }
//}
