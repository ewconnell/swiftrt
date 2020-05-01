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
/// SNLM
public struct Embedding<Element> : Module
where Element: DifferentiableElement & BinaryFloatingPoint
{
    /// A learnable lookup table that maps vocabulary indices to their dense vector representations.
    public var embeddings: TensorR2<Element>
    
    /// Creates an `Embedding` layer with randomly initialized embeddings of shape
    /// `(vocabularySize, embeddingSize)` so that each vocabulary index is given a vector
    /// representation.
    ///
    /// - Parameters:
    ///   - vocabularySize: The number of distinct indices (words) in the vocabulary. This number
    ///     should be the largest integer index plus one.
    ///   - embeddingSize: The number of entries in a single embedding vector representation.
    ///   - embeddingsInitializer: Initializer to use for the embedding parameters.
    public init(
        vocabularySize: Int,
        embeddingSize: Int,
        embeddingsInitializer: ParameterInitializer<Shape2, Element> = { TensorR2(randomUniform: $0) }
    ) {
        precondition(vocabularySize > 0, "The vocabulary size must be greater than 0.")
        precondition(embeddingSize > 0, "The embedding size must be greater than 0.")
        self.init(embeddings: embeddingsInitializer([vocabularySize, embeddingSize]))
    }
    
    /// Creates an `Embedding` layer from the provided embeddings. Useful for introducing
    /// pretrained embeddings into a model.
    ///
    /// - Parameter embeddings: The pretrained embeddings table.
    public init(embeddings: TensorR2<Element>) {
        self.embeddings = embeddings
    }
    
    /// Returns an output by replacing each index in the input with corresponding dense vector representation.
    ///
    /// - Parameter
    ///   - input: The indices that will be mapped to their vector representations.
    /// - Returns: The tensor created by replacing input indices with their vector representations.
    @differentiable(wrt: self)
    public func callAsFunction(_ input: TensorR1<DeviceIndex>) -> TensorR2<Element> {
        embeddings.gathering(indices: input)
    }
}
