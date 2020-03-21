//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Numerics

// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// V2 paper
// "Bag of Tricks for Image Classification with Convolutional Neural Networks"
// Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
// https://arxiv.org/abs/1812.01187

// A convolution and batchnorm layer
public struct DRNV2_ConvBN<T, F> where
    T: DifferentiableTensorView,
    T.Element: ScalarElement & BinaryFloatingPoint,
    F: TensorView, F.Bounds == T.Bounds,
    F.Element: ScalarElement & Real & BinaryFloatingPoint
{
    public var conv: Convolution<T, F>
    public var norm: BatchNorm<T>
    public let isLast: Bool

    public init(
        inFilters: Int,
        outFilters: Int,
        kernelSize: Int = 1,
        stride: Int = 1,
        padding: Padding = .same,
        isLast: Bool = false
    ) {
        // setup filter shape
        var filterShape = T.Bounds(repeating: kernelSize)
        filterShape[T.rank - 2] = inFilters
        filterShape[T.rank - 1] = outFilters

        // bias is not used
        self.conv = Convolution(filterShape: filterShape,
                                stride: stride,
                                padding: padding,
                                filterInitializer: glorotUniform())
        self.isLast = isLast
        if isLast {
            //Initialize the last BatchNorm layer to scale zero
            self.norm = BatchNorm(
                 axis: -1,
                 momentum: 0.9,
                 offset: Vector<T.Element>(zeros: (outFilters)),
                 scale: Vector<T.Element>(zeros: (outFilters)),
                 epsilon: 1e-5,
                 runningMean: Vector(T.Element.zero),
                 runningVariance: Vector(T.Element.one))
        } else {
            self.norm = BatchNorm(featureCount: outFilters,
                                  momentum: 0.9, epsilon: 1e-5)
        }
    }

    public func callAsFunction(_ input: T) -> T {
        input
//        let convResult = input.sequenced(through: conv, norm)
//        return isLast ? convResult : relu(convResult)
    }
}
