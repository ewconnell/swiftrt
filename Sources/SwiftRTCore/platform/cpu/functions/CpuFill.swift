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

extension CpuFunctions where Self: CpuMapOps {
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(_ result: inout Tensor<S,E>, with element: E)
    where S: TensorShape
    {
        generatorOp(&result) { element }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E,B>(
        _ result: inout Tensor<S,E>,
        with range: Range<B>
    ) where S: TensorShape, E: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    {
        mapOp(range.lazy.map { E(exactly: $0)! }, &result) { $0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func eye<S,E>(_ result: inout Tensor<S,E>, offset: Int)
    where S: TensorShape, E: Numeric
    {
        assert(!result.isSequential)
        generatorOp(&result) { 0 }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E,
        _ upper: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let scale = Double(upper - lower) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        
        generatorOp(&result) {
            E(Double(generator.next()) * scale) + lower
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E,
        _ standardDeviation: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        
        generatorOp(&result) {
            E(Double(generator.next()) * scale) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        
        generatorOp(&result) {
            E(Double(generator.next()) * scale) + mean.element
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E,
        _ standardDeviation: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let std2x = standardDeviation * 2
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        
        generatorOp(&result) {
            let a = Double(generator.next()) * scale
            return E(a).clamped(to: -std2x...std2x) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let std2x = standardDeviation.element * 2
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        
        generatorOp(&result) {
            let a = Double(generator.next()) * scale
            return E(a).clamped(to: -std2x...std2x) + mean.element
        }
    }
}
