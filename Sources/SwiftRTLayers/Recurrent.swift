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
    public init(input: Input, state: State) {
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
    public init(output: Output, state: State) {
        self.output = output
        self.state = state
    }
}

extension RNNCellOutput: EuclideanDifferentiable
where Output: EuclideanDifferentiable, State: EuclideanDifferentiable {}

//==============================================================================
/// RecurrentLayerCell
/// A recurrent neural network cell.
public protocol RecurrentLayerCell: Layer
where Input == RNNCellInput<TimeStepInput, State>,
      Output == RNNCellOutput<TimeStepOutput, State>
{
    /// The input at a time step.
    associatedtype TimeStepInput: Differentiable
    /// The output at a time step.
    associatedtype TimeStepOutput: Differentiable
    /// The state that may be preserved across time steps.
    associatedtype State: Differentiable
    
    /// Returns a zero-valued state with a shape compatible to the input
    func zeroState(for input: TimeStepInput) -> State
}

extension RecurrentLayerCell {
    //--------------------------------------------------------------------------
    /// Returns the new state obtained from applying the recurrent layer
    /// cell to the input at the current time step and the previous state.
    ///
    /// - Parameters:
    ///   - timeStepInput: The input at the current time step.
    ///   - previousState: The previous state of the recurrent layer cell.
    /// - Returns: The output.
    @differentiable
    @inlinable public func callAsFunction(
        input: TimeStepInput,
        state: State
    ) -> RNNCellOutput<TimeStepOutput, State> {
        self(RNNCellInput(input: input, state: state))
    }
    
    @differentiable
    @inlinable public func call(
        input: TimeStepInput,
        state: State
    ) -> RNNCellOutput<TimeStepOutput, State> {
        self(RNNCellInput(input: input, state: state))
    }
}

//==============================================================================
/// A basic RNN cell.
public struct BasicRNNCell<Element>: RecurrentLayerCell
where Element: DifferentiableElement & Real & BinaryFloatingPoint
{
    public var weight: TensorR2<Element>
    public var bias: TensorR2<Element>

    // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after SR-10697 is fixed.
    public struct State:
    Equatable, Differentiable, VectorProtocol, KeyPathIterable
    {
        @inlinable public func adding(_ x: Element) -> Self {
            State(value + x)
        }
        
        @inlinable public func subtracting(_ x: Element) -> Self {
            State(value - x)
        }
        
        @inlinable public func scaled(by scalar: Element) -> Self {
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
    
    //--------------------------------------------------------------------------
    /// Creates a `BasicRNNCell` with the specified input size and
    /// hidden state size.
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
        let weightShape = Shape2(inputSize + hiddenSize, hiddenSize)
        self.weight = TensorR2(glorotUniform: weightShape, seed: seed)
        // TODO: not sure if row or column
        self.bias = TensorR2(zeros: Shape2(1, hiddenSize))
    }
    
    //--------------------------------------------------------------------------
    /// Returns a zero-valued state with shape compatible with the provided input.
    @inlinable public func zeroState(
        for input: TensorR2<Element>
    ) -> State {
        State(TensorR2<Element>(zeros: Shape2(input.shape[0], weight.shape[1])))
    }
    
    //--------------------------------------------------------------------------
    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The hidden state.
    @differentiable
    @inlinable public func callAsFunction(_ input: Input) -> Output {
        let concatenatedInput = input.input
                .concatenated(with: input.state.value, alongAxis: 1)
        let newState = State(tanh(matmul(concatenatedInput, weight) + bias))
        return Output(output: newState, state: newState)
    }
}

//==============================================================================
/// An LSTM cell.
public struct LSTMCell<Element>: RecurrentLayerCell
where Element: DifferentiableElement & Real & BinaryFloatingPoint
{
    // types
    public typealias TimeStepInput = TensorR2<Element>
    public typealias TimeStepOutput = State
    public typealias Input = RNNCellInput<TimeStepInput, State>
    public typealias Output = RNNCellOutput<TimeStepOutput, State>
    public enum Part: Int, CaseIterable { case input, update, forget, output }
    
    // properties
    public var fusedWeight: TensorR2<Element>
    public var fusedBias: TensorR1<Element>
    @noDerivative public let hiddenSize: Int
    
    //--------------------------------------------------------------------------
    /// Creates a `LSTMCell` with the specified input size and hidden state size.
    ///
    /// - Parameters:
    ///   - inputSize: The number of features in 2-D input tensors.
    ///   - hiddenSize: The number of features in 2-D hidden states.
    public init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        self.fusedBias = Tensor(zeros: (4 * hiddenSize))
        self.fusedWeight =
            Tensor(glorotUniform: (inputSize + hiddenSize, 4 * hiddenSize))
    }
    
    //--------------------------------------------------------------------------
    /// part
    /// used to access parts of the fused weights
    @differentiable
    @inlinable public func part(_ i: Part, of fused: TensorR2<Element>)
    -> TensorR2<Element>
    {
        fused[0..., (i.rawValue * hiddenSize)..<((i.rawValue + 1) * hiddenSize)]
    }
    
    /// part
    /// used to access parts of the fused bias
    @differentiable
    @inlinable public func part(_ i: Part, of fused: TensorR1<Element>)
    -> TensorR1<Element>
    {
        fused[(i.rawValue * hiddenSize)..<((i.rawValue + 1) * hiddenSize)]
    }
    
    //--------------------------------------------------------------------------
    public struct State:
        Equatable, Differentiable, VectorProtocol, KeyPathIterable
    {
        // TODO: Verify that is is correct and find out why I had to implement it
        public func adding(_ x: Element) -> Self {
            State(cell: cell + x, hidden: hidden + x)
        }
        
        public func subtracting(_ x: Element) -> Self {
            State(cell: cell - x, hidden: hidden - x)
        }
        
        public func scaled(by scalar: Element) -> Self {
            State(cell: cell * scalar, hidden: hidden * scalar)
        }        
        
        public typealias VectorSpaceScalar = Element
        public var cell: TensorR2<Element>
        public var hidden: TensorR2<Element>
        
        @differentiable
        @inlinable public init(cell: TensorR2<Element>, hidden: TensorR2<Element>) {
            self.cell = cell
            self.hidden = hidden
        }
    }
    
    //--------------------------------------------------------------------------
    /// Returns a zero-valued state with shape compatible with the provided input.
    @inlinable public func zeroState(for input: TensorR2<Element>) -> State {
        let shape = Shape2(input.shape[0], hiddenSize)
        return State(cell: Tensor(zeros: shape), hidden: Tensor(zeros: shape))
    }
    
    //--------------------------------------------------------------------------
    /// Returns the output obtained from applying the layer to the given input
    /// - Parameter input: The input to the layer.
    /// - Returns: The hidden state.
    @differentiable
    @inlinable public func callAsFunction(_ input: Input) -> Output {
        let gateInput = concatenate(input.input, input.state.hidden, axis: 1)
        let fused = matmul(gateInput, fusedWeight, bias: fusedBias)
        let inputGate = sigmoid(part(.input, of: fused))
        let updateGate = tanh(part(.update, of: fused))
        let forgetGate = sigmoid(part(.forget, of: fused))
        let outputGate = sigmoid(part(.output, of: fused))
        let newCellState = input.state.cell * forgetGate + inputGate * updateGate
        let newHiddenState = tanh(newCellState) * outputGate
        let newState = State(cell: newCellState, hidden: newHiddenState)
        return Output(output: newState, state: newState)
    }
}

//==============================================================================
/// An GRU cell.
public struct GRUCell<Element>: RecurrentLayerCell
where Element: DifferentiableElement & Real & BinaryFloatingPoint
{
    public var updateWeight1, updateWeight2: TensorR2<Element>
    public var resetWeight1, resetWeight2: TensorR2<Element>
    public var outputWeight1, outputWeight2: TensorR2<Element>
    public var updateBias, outputBias, resetBias: TensorR1<Element>
    
    @noDerivative public var stateShape: Shape2 {
        Shape2(1, updateWeight1.shape[0])
    }
    
    //--------------------------------------------------------------------------
    /// Returns a zero-valued state with shape compatible with the provided input.
    public func zeroState(
        for input: TensorR2<Element>
    ) -> State {
        State(hidden: TensorR2<Element>(zeros: stateShape))
    }
    
    public typealias TimeStepInput = TensorR2<Element>
    public typealias TimeStepOutput = State
    public typealias Input = RNNCellInput<TimeStepInput, State>
    public typealias Output = RNNCellOutput<TimeStepOutput, State>
    
    /// Creates a `GRUCell` with the specified input size and hidden state size.
    ///
    /// - Parameters:
    ///   - inputSize: The number of features in 2-D input tensors.
    ///   - hiddenSize: The number of features in 2-D hidden states.
    public init(
        inputSize: Int,
        hiddenSize: Int,
        weightInitializer: ParameterInitializer<Shape2,Element> = glorotUniform(),
        biasInitializer: ParameterInitializer<Shape1,Element> = zeros()
    ) {
        let gateWeightShape = Shape2(inputSize, 1)
        let gateBiasShape = Shape1(hiddenSize)
        self.updateWeight1 = weightInitializer(gateWeightShape)
        self.updateWeight2 = weightInitializer(gateWeightShape)
        self.updateBias = biasInitializer(gateBiasShape)
        self.resetWeight1 = weightInitializer(gateWeightShape)
        self.resetWeight2 = weightInitializer(gateWeightShape)
        self.resetBias = biasInitializer(gateBiasShape)
        self.outputWeight1 = weightInitializer(gateWeightShape)
        self.outputWeight2 = weightInitializer(gateWeightShape)
        self.outputBias = biasInitializer(gateBiasShape)
    }
    
    // TODO(TF-507): Revert to `typealias State = Tensor<Scalar>` after
    // SR-10697 is fixed.
    public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable {
        public func adding(_ x: Element) -> Self {
            State(hidden: hidden + x)
        }
        
        public func subtracting(_ x: Element) -> Self {
            State(hidden: hidden - x)
        }
        
        public func scaled(by scalar: Element) -> Self {
            State(hidden: hidden * scalar)
        }
        
        public typealias VectorSpaceScalar = Element
        public var hidden: TensorR2<Element>
        
        @differentiable
        public init(hidden: TensorR2<Element>) {
            self.hidden = hidden
        }
    }
    
    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The hidden state.
    @differentiable
    public func callAsFunction(_ input: Input) -> Output {
        let resetGate = sigmoid(
            matmul(input.input, resetWeight1) +
                matmul(input.state.hidden, resetWeight2, bias: resetBias))
        
        let updateGate = sigmoid(
                matmul(input.input, updateWeight1) +
                    matmul(input.state.hidden, updateWeight2, bias: updateBias))
        
        let outputGate = tanh(
                matmul(input.input, outputWeight1, bias: outputBias) +
                    matmul(resetGate * input.state.hidden, outputWeight2))
        
        let updateHidden = (1 - updateGate) * input.state.hidden
        let updateOutput = (1 - updateGate) * outputGate
        let newState = State(hidden: updateHidden + updateOutput)
        return Output(output: newState, state: newState)
    }
}

//==============================================================================
public struct RecurrentLayer<Cell: RecurrentLayerCell>: Layer {
    public typealias Input = [Cell.TimeStepInput]
    public typealias Output = [Cell.TimeStepOutput]
    
    public var cell: Cell
    
    public init(_ cell: @autoclosure () -> Cell) {
        self.cell = cell()
    }
    
    @differentiable(wrt: (self, inputs, initialState))
    public func callAsFunction(
        _ inputs: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> [Cell.TimeStepOutput] {
        if inputs.isEmpty { return [Cell.TimeStepOutput]() }
        var currentHiddenState = initialState
        var timeStepOutputs: [Cell.TimeStepOutput] = []
        for timeStepInput in inputs {
            let output = cell(input: timeStepInput, state: currentHiddenState)
            currentHiddenState = output.state
            timeStepOutputs.append(output.output)
        }
        return timeStepOutputs
    }
    
    @differentiable(wrt: (self, inputs, initialState))
    public func call(
        _ inputs: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> [Cell.TimeStepOutput] {
        callAsFunction(inputs, initialState: initialState)
    }
    
    @usableFromInline
    @derivative(of: callAsFunction, wrt: (self, inputs, initialState))
    internal func _vjpCallAsFunction(
        _ inputs: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> (
        value: [Cell.TimeStepOutput],
        pullback: (Array<Cell.TimeStepOutput>.TangentVector)
            -> (TangentVector, Array<Cell.TimeStepInput>.TangentVector, Cell.State.TangentVector)
    ) {
        let timeStepCount = inputs.count
        var currentHiddenState = initialState
        var timeStepOutputs: [Cell.TimeStepOutput] = []
        timeStepOutputs.reserveCapacity(timeStepCount)
        var backpropagators: [Cell.Backpropagator] = []
        backpropagators.reserveCapacity(timeStepCount)
        for timestep in inputs {
            let (output, backpropagator) = cell.appliedForBackpropagation(
                    to: .init(input: timestep, state: currentHiddenState))
            currentHiddenState = output.state
            timeStepOutputs.append(output.output)
            backpropagators.append(backpropagator)
        }
        return (
            timeStepOutputs,
            { 𝛁outputs in
                precondition(
                    𝛁outputs.base.count == timeStepCount,
                    "The number of output gradients must equal the number of time steps")
                var 𝛁cell = Cell.TangentVector.zero
                var 𝛁state = Cell.State.TangentVector.zero
                var reversed𝛁inputs: [Cell.TimeStepInput.TangentVector] = []
                reversed𝛁inputs.reserveCapacity(timeStepCount)
                for (𝛁output, backpropagator) in zip(𝛁outputs.base, backpropagators).reversed() {
                    let (new𝛁cell, 𝛁input) = backpropagator(.init(output: 𝛁output, state: 𝛁state))
                    𝛁cell += new𝛁cell
                    𝛁state = 𝛁input.state
                    reversed𝛁inputs.append(𝛁input.input)
                }
                return (.init(cell: 𝛁cell), .init(Array(reversed𝛁inputs.reversed())), 𝛁state)
            }
        )
    }
    
    @differentiable
    public func callAsFunction(_ inputs: [Cell.TimeStepInput]) -> [Cell.TimeStepOutput] {
        let initialState = withoutDerivative(at: cell.zeroState(for: inputs[0]))
        return self(inputs, initialState: initialState)
    }
    
    @differentiable(wrt: (self, inputs, initialState))
    public func lastOutput(
        from inputs: [Cell.TimeStepInput],
        initialState: Cell.State
    ) -> Cell.TimeStepOutput {
        precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
        return self(inputs, initialState: initialState)[withoutDerivative(at: inputs.count - 1)]
    }
    
    @differentiable(wrt: (self, inputs))
    public func lastOutput(from inputs: [Cell.TimeStepInput]) -> Cell.TimeStepOutput {
        precondition(!inputs.isEmpty, "'inputs' must be non-empty.")
        let initialState = withoutDerivative(at: cell.zeroState(for: inputs[0]))
        return lastOutput(from: inputs, initialState: initialState)
    }
}

//==============================================================================
extension RecurrentLayer: Equatable where Cell: Equatable {}
extension RecurrentLayer: AdditiveArithmetic where Cell: AdditiveArithmetic {}

public typealias BasicRNN<Element> = RecurrentLayer<BasicRNNCell<Element>>
    where Element: Real & BinaryFloatingPoint & DifferentiableElement

public typealias LSTM<Element> = RecurrentLayer<LSTMCell<Element>>
    where Element: Real & BinaryFloatingPoint & DifferentiableElement

public typealias GRU<Element> = RecurrentLayer<GRUCell<Element>>
    where Element: Real & BinaryFloatingPoint & DifferentiableElement
