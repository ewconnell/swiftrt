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
import SwiftRTCore
import Numerics

//==============================================================================
/// RNNCellInput
/// An input to a recurrent neural network
public struct RNNCellInput<Input: Differentiable, State: Differentiable>:
    Differentiable
{
    /// The input at the current time step.
    public var input: Input
    /// The previous state.
    public var state: State
    
    @differentiable
    @inlinable public init(input: Input, state: State) {
        self.input = input
        self.state = state
    }
}

extension RNNCellInput: EuclideanDifferentiable
where Input: EuclideanDifferentiable, State: EuclideanDifferentiable {}

//==============================================================================
/// RNNCellOutput
/// An output from a recurrent neural network
public struct RNNCellOutput<Output: Differentiable, State: Differentiable>:
    Differentiable
{
    /// The output at the current time step.
    public var output: Output
    /// The current state.
    public var state: State
    
    @differentiable
    @inlinable public init(output: Output, state: State) {
        self.output = output
        self.state = state
    }
}

extension RNNCellOutput: EuclideanDifferentiable
    where Output: EuclideanDifferentiable, State: EuclideanDifferentiable {}

//==============================================================================
/// RNNCell
/// A recurrent neural network cell.
public protocol RNNCell: Layer
where Input == RNNCellInput<TimeStepInput, State>,
      Output == RNNCellOutput<TimeStepOutput, State>
{
    /// The input at a time step.
    associatedtype TimeStepInput: Differentiable
    /// The output at a time step.
    associatedtype TimeStepOutput: Differentiable
    /// The state that may be preserved across time steps.
    associatedtype State: Differentiable
    
    /// Returns a zero-valued state with shape compatible with the provided input.
    func zeroState(for input: TimeStepInput) -> State
}

public extension RNNCell
{
    /// Returns the new state obtained from applying the RNN cell to the input at the current time
    /// step and the previous state.
    ///
    /// - Parameters:
    ///   - timeStepInput: The input at the current time step.
    ///   - previousState: The previous state of the RNN cell.
    /// - Returns: The output.
    @differentiable
    @inlinable func callAsFunction(
        input: TimeStepInput,
        state: State
    ) -> RNNCellOutput<TimeStepOutput, State> {
        self(RNNCellInput(input: input, state: state))
    }
    
    @differentiable
    @inlinable func call(input: TimeStepInput, state: State)
    -> RNNCellOutput<TimeStepOutput, State>
    {
        self(RNNCellInput(input: input, state: state))
    }
}

//==============================================================================
/// SimpleRNNCell
/// A simple RNN cell.
public struct SimpleRNNCell<Element>: RNNCell
where Element: DifferentiableElement & Real & BinaryFloatingPoint
{
    public var weight: TensorR2<Element>
    public var bias: TensorR2<Element>
    
    // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after SR-10697 is fixed.
    public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable {
        public func adding(_ x: Element) -> SimpleRNNCell<Element>.State {
            State(value + x)
        }
        
        public func subtracting(_ x: Element) -> SimpleRNNCell<Element>.State {
            State(value - x)
        }
        
        public func scaled(by scalar: Element) -> SimpleRNNCell<Element>.State {
            State(value * scalar)
        }
        
        public typealias VectorSpaceScalar = Element
        public var value: TensorR2<Element>
        @differentiable
        @inlinable public init(_ value: TensorR2<Element>) {
            self.value = value
        }
    }
    
    public typealias TimeStepInput = TensorR2<Element>
    public typealias TimeStepOutput = State
    public typealias Input = RNNCellInput<TimeStepInput, State>
    public typealias Output = RNNCellOutput<TimeStepOutput, State>
    
    /// Creates a `SimpleRNNCell` with the specified input size and hidden state size.
    ///
    /// - Parameters:
    ///   - inputSize: The number of features in 2-D input tensors.
    ///   - hiddenSize: The number of features in 2-D hidden states.
    ///   - seed: The random seed for initialization. The default value is random.
    @inlinable public init(
        inputSize: Int,
        hiddenSize: Int,
        seed: RandomSeed = Context.randomSeed
    ) {
        let concatenatedInputSize = inputSize + hiddenSize
        let weightShape = Shape2(concatenatedInputSize, hiddenSize)
        self.weight = TensorR2(glorotUniform: weightShape, seed: seed)
        // TODO: not sure if row or column
        self.bias = TensorR2(zeros: Shape2(1, hiddenSize))
    }
    
    /// Returns a zero-valued state with shape compatible with the provided input.
    @inlinable public func zeroState(for input: TensorR2<Element>) -> State {
        State(TensorR2<Element>(zeros: Shape2(input.shape[0], weight.shape[1])))
    }
    
    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The hidden state.
    @differentiable
    @inlinable public func callAsFunction(_ input: Input) -> Output {
        // TODO: this concat is really inefficient
        let concatenatedInput = input.input.concatenated(with: input.state.value, alongAxis: 1)
        let newState = State(tanh(matmul(concatenatedInput, weight) + bias))
        return Output(output: newState, state: newState)
    }
}

