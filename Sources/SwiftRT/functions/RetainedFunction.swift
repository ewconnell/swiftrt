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

//==============================================================================
/// RetainedInputFunction
/// adopted by objects that have a differentiable output
public protocol RetainedInputFunction:
    EuclideanDifferentiable, KeyPathIterable where
    TangentVector: VectorProtocol & ElementaryFunctions &
    PointwiseMultiplicative & KeyPathIterable
{
    /// The function input type.
    associatedtype Input
    /// The function output type.
    associatedtype Output: Differentiable
    /// callAsFunction
    /// - Parameter input: The function input
    /// - Returns: The function output
    @differentiable(wrt: self)
    func callAsFunction(_ input: Input) -> Output
}

//==============================================================================
/// RetainedFunction
/// Types that conform to `RetainedFunction` represent functions that
/// map inputs to outputs. They may have an internal state represented
/// by parameters such as weights or device specific resources.
///
/// `RetainedFunction` instances define a differentiable
/// `callAsFunction(_:)` method for mapping inputs to outputs
public protocol RetainedFunction: RetainedInputFunction
    where Input: Differentiable
{
    /// callAsFunction
    /// - Parameter input: The function input
    /// - Returns: The function output
    @differentiable
    func callAsFunction(_ input: Input) -> Output
}
