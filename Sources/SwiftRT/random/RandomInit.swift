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
// Random initializers
public extension TensorView where Element: Numeric
{
    /// Creates a tensor with the specified shape, randomly sampling scalar
    /// values from a uniform distribution between `lowerBound` and `upperBound`
    ///
    /// - Parameters:
    ///   - bounds: The dimensions of the tensor
    ///   - lowerBound: The lower bound of the distribution
    ///   - upperBound: The upper bound of the distribution
    ///   - seed: The seed value
    init(randomUniform bounds: Bounds,
         lowerBound: Self,
         upperBound: Self,
         seed: RandomSeed = Context.randomSeed)
    {
        fatalError()
    }
}

