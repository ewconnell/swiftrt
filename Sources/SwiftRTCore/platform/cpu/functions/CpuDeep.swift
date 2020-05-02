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

extension CpuFunctions {
    //==========================================================================
//    public func activation<S,E>(
//        mode: ActivationType,
//        nan: NanPropagation,
//        reluCeiling: Double = 0) throws -> DeviceActivation<S,E>
//    where S: TensorShape, E: ScalarElement & BinaryFloatingPoint
//    {
//        fatalError("cpu not implemented")
//    }

    //==========================================================================
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
        fatalError("cpu convolution not implemented")
    }
}
