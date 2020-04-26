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
///// BatchNorm
//public struct BatchNorm<T> where
//    T: DifferentiableTensorView, T.Element: ScalarElement & BinaryFloatingPoint
//{
//    /// The feature dimension.
//    @noDerivative public let axis: Int
//    /// The momentum for the running mean and running variance.
//    @noDerivative public let momentum: T.Element
//    /// The offset value, also known as beta.
//    public var offset: Vector<T.Element>
//    /// The scale value, also known as gamma.
//    public var scale: Vector<T.Element>
//    /// The variance epsilon value.
//    @noDerivative public let epsilon: T.Element
//    /// The running mean.
//    @noDerivative public var runningMean: Parameter<Vector<T.Element>>
//    /// The running variance.
//    @noDerivative public var runningVariance: Parameter<Vector<T.Element>>
//
//    //--------------------------------------------------------------------------
//    /// Creates a batch normalization layer.
//    ///
//    /// - Parameters:
//    ///   - axis: The axis that should not be normalized (typically the feature axis).
//    ///   - momentum: The momentum for the moving average.
//    ///   - offset: The offset to be added to the normalized tensor.
//    ///   - scale: The scale to multiply the normalized tensor by.
//    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
//    ///   - runningMean: The running mean.
//    ///   - runningVariance: The running variance.
//    public init(
//        axis: Int,
//        momentum: T.Element,
//        offset: Vector<T.Element>,
//        scale: Vector<T.Element>,
//        epsilon: T.Element,
//        runningMean: Vector<T.Element>,
//        runningVariance: Vector<T.Element>
//    ) {
//        precondition(offset.count == scale.count,
//                     "The offset and the scale must have same count")
//        self.axis = axis
//        self.momentum = momentum
//        self.offset = offset
//        self.scale = scale
//        self.epsilon = epsilon
//        self.runningMean = Parameter(runningMean)
//        self.runningVariance = Parameter(runningVariance)
//    }
//
//    /// Creates a batch normalization layer.
//    ///
//    /// - Parameters:
//    ///   - featureCount: The number of features.
//    ///   - axis: The axis that should be normalized (typically the features axis).
//    ///   - momentum: The momentum for the moving average.
//    ///   - epsilon: A small scalar added to the denominator to improve numerical stability.
//    public init(
//        featureCount: Int,
//        axis: Int = -1,
//        momentum: T.Element = 0.99,
//        epsilon: T.Element = 0.001
//    ) {
//        self.init(
//            axis: axis,
//            momentum: momentum,
//            offset: Vector<T.Element>(zeros: (featureCount)),
//            scale: Vector<T.Element>(ones: (featureCount)),
//            epsilon: epsilon,
//            runningMean: Vector(T.Element.zero),
//            runningVariance: Vector(T.Element.one))
//    }
//    /// Returns the output obtained from applying the layer to the given input.
//    ///
//    /// - Parameter input: The input to the layer.
//    /// - Returns: The output.
//    @differentiable
//    public func callAsFunction(_ input: T) -> T {
//        input
////        let positiveAxis = (input.rank + axis) % input.rank
////        precondition(input.shape[positiveAxis] == offset.shape[0],
////                     "The number of features of the input and the offset doesn't match.")
////        var offset = self.offset
////        var scale = self.scale
////        if positiveAxis != input.rank - 1 {
////            var broadcastShape = TensorShape([Int](repeating: 1, count: input.rank))
////            broadcastShape[positiveAxis] = input.shape[positiveAxis]
////            offset = offset.reshaped(to: broadcastShape)
////            scale = scale.reshaped(to: broadcastShape)
////        }
////        switch Context.local.learningPhase {
////        case .training:
////          var normalizedAxes = Array(0..<input.rank)
////          normalizedAxes.remove(at: positiveAxis)
////          let moments = input.moments(alongAxes: normalizedAxes)
////          let decayMomentum = Tensor(1 - momentum, on: input.device)
////          runningMean.value += (moments.mean - runningMean.value) * decayMomentum
////          runningVariance.value += (moments.variance - runningVariance.value) * decayMomentum
////          let inv = rsqrt(moments.variance + Tensor(epsilon, on: input.device)) * scale
////          return (input - moments.mean) * inv + offset
////        case .inference:
////          let inv = rsqrt(runningVariance.value + Tensor(epsilon, on: input.device)) * scale
////          return (input - runningMean.value) * inv + offset
////        }
//    }
//}
