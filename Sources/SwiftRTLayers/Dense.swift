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
import SwiftRTCore

//==============================================================================
///
public struct Dense<S,E> : Layer
where S: TensorShape,
      E: StorageElement,
      E.Value: DifferentiableNumeric & BinaryFloatingPoint
{
    /// The element-wise activation function.
    @noDerivative public let activation: ActivationType
    /// The execution plan key
    @noDerivative public var plan: Int?
    /// The weights
    public var weight: Tensor<S,E>
    /// The bias
    public var bias: Tensor<S,E>
    
    //--------------------------------------------------------------------------
    @differentiable
    public func callAsFunction(_ input: Tensor<S,E>) -> Tensor<S,E> {
        input
    }
}

public extension Dense where S == Shape2 {
    @inlinable init(
        weight: TensorR2<E>,
        bias: TensorR1<E>,
        activation: ActivationType
    ) {
        assert(weight.order == bias.order &&
                bias.shape[0] == weight.shape[1])
        self.activation = activation
        self.weight = weight
        self.bias = Tensor<S,E>(reshaping: bias, to: weight.shape,
                                order: weight.order)
    }
}
public extension Dense where S == Shape3 {

    @inlinable init(
        weight: TensorR3<E>,
        bias: TensorR2<E>,
        activation: ActivationType
    ) {
        assert(weight.order == bias.order &&
                bias.shape[0] == weight.shape[1] &&
                bias.shape[1] == weight.shape[2])
        self.activation = activation
        self.weight = weight
        self.bias = Tensor<S,E>(reshaping: bias, to: weight.shape,
                                order: weight.order)
    }
}

//extension Dense {
//    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
//    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
//    /// the bias vector is created with shape `[outputSize]`.
//    ///
//    /// - Parameters:
//    ///   - inputSize: The dimensionality of the input space.
//    ///   - outputSize: The dimensionality of the output space.
//    ///   - activation: The activation function to use. The default value is `identity(_:)`.
//    ///   - weightInitializer: Initializer to use for `weight`.
//    ///   - biasInitializer: Initializer to use for `bias`.
//    public init(
//        inputSize: Int,
//        outputSize: Int,
//        activation: @escaping Activation = identity,
//        useBias: Bool = true,
//        weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
//        biasInitializer: ParameterInitializer<Scalar> = zeros()
//    ) {
//        self.init(
//            weight: weightInitializer([inputSize, outputSize]),
//            bias: useBias ? biasInitializer([outputSize]) : nil,
//            activation: activation)
//    }
//}
