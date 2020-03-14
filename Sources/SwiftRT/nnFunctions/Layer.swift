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

// Note: `Module` and `Layer` protocol definitions are adapted and simplified from:
// https://github.com/tensorflow/swift-apis/blob/master/Sources/TensorFlow/Layer.swift

public protocol Module: Differentiable, KeyPathIterable
    where TangentVector: KeyPathIterable {
    /// The input type of the layer.
    associatedtype Input
    /// The output type of the layer.
    associatedtype Output: Differentiable

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable(wrt: self)
    func callAsFunction(_ input: Input) -> Output
}

/// A neural network layer.
///
/// Types that conform to `Layer` represent functions that map inputs to outputs. They may have an
/// internal state represented by parameters, such as weight tensors.
///
/// `Layer` instances define a differentiable `callAsFunction(_:)` method for mapping inputs to
/// outputs.
public protocol Layer: Module where Input: Differentiable {
    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    func callAsFunction(_ input: Input) -> Output
}

public extension Layer {
    @differentiable
    func call(_  input: Input) -> Output {
        callAsFunction(input)
    }
}

/// An empty struct representing empty `TangentVector`s for parameterless layers.
public struct EmptyTangentVector: EuclideanDifferentiable, VectorProtocol, ElementaryFunctions,
                                  PointwiseMultiplicative, KeyPathIterable {
    public typealias VectorSpaceScalar = Float
    public typealias TangentVector = Self

    public init() {}

    public func adding(_ x: Float) -> EmptyTangentVector { self }
    public mutating func add(_ x: Float) {}
    public func subtracting(_ x: Float) -> EmptyTangentVector { self }
    public mutating func subtract(_ x: Float) {}
    public func scaled(by scalar: Float) -> EmptyTangentVector { self }
    public mutating func scale(by scalar: Float) {}
}

/// A parameterless neural network layer.
///
/// The `TangentVector` of parameterless layers is always `EmptyTangentVector`.
public protocol ParameterlessLayer: Layer where TangentVector == EmptyTangentVector {
    @differentiable
    func callAsFunction(_ input: Input) -> Output
}

public extension ParameterlessLayer {
    mutating func move(along direction: EmptyTangentVector) {}
    var differentiableVectorView: EmptyTangentVector { EmptyTangentVector() }
}

public extension Layer {
    /// Returns the inference output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The inference output.
    func inferring(from input: Input) -> Output {
        Context.whileInferring { self(input) }
    }

    // TODO(TF-433, SR-11882): Remove this custom derivative when
    // differentiation supports `rethrows` functions and currying.
    @derivative(of: inferring(from:))
    @usableFromInline
    internal func _vjpInferring(from input: Input)
        -> (value: Output, pullback: (Output.TangentVector)
            -> (TangentVector, Input.TangentVector)) {
        Context.whileInferring {
            let (output, pullback) = appliedForBackpropagation(to: input)
            return (output, { v in pullback(v) })
        }
    }

    typealias Backpropagator = (_ direction: Output.TangentVector)
        -> (layerGradient: TangentVector, inputGradient: Input.TangentVector)

    /// Returns the inference output and the backpropagation function obtained from applying the
    /// layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: A tuple containing the output and the backpropagation function. The
    ///   backpropagation function (a.k.a. backpropagator) takes a direction vector and returns the
    ///   gradients at the layer and at the input, respectively.
    func appliedForBackpropagation(to input: Input)
        -> (output: Output, backpropagator: Backpropagator) {
        let (out, pullback) = Swift.valueWithPullback(at: self, input) { layer, input in
            return layer(input)
        }
        return (out, pullback)
    }
}
