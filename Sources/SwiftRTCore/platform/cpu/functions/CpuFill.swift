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
// DeviceQueue functions with default cpu delegation
extension DeviceQueue where Self: CpuFunctions
{
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E: StorageElement>(
        _ result: inout Tensor<S,E>,
        with element: E.Value
    ) where S: TensorShape { cpu_fill(&result, with: element) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E,B>(
        _ result: inout Tensor<S,E>,
        with range: Range<B>
    ) where S: TensorShape, E: StorageElement, E.Value: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    { cpu_fill(&result, with: range) }
    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ result: inout Tensor<S,E>,
        offset: Int
    ) where S: TensorShape, E.Value: Numeric { cpu_eye(&result, offset: offset) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomUniform: &result, lower, upper, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomNormal: &result, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomNormal: &result, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomTruncatedNormal: &result, mean, standardDeviation, seed) }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable func cpu_eye<S,E>(
        _ result: inout Tensor<S,E>,
        offset: Int
    ) where S: TensorShape, E.Value: Numeric {
        mapOp("cpu_eye", &result) { 0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        _ result: inout Tensor<S,E>,
        with element: E.Value
    ) where S: TensorShape {
        mapOp("cpu_fill element", &result) { element }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E,B>(
        _ result: inout Tensor<S,E>,
        with range: Range<B>
    ) where S: TensorShape, E.Value: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    {
        mapOp("cpu_fill range", range.lazy.map { E.Value(exactly: $0)! }, &result)
    }
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    {
        let scale = Double(upper - lower) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp("cpu_fill randomUniform", &result) {
            E.Value(Double(generator.next()) * scale) + lower
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    {
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp("cpu_fill randomNormal", &result) {
            E.Value(Double(generator.next()) * scale) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func cpu_fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp("cpu_fill randomNormal", &result) {
            E.Value(Double(generator.next()) * scale) + mean.element
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    {
        let std2x = standardDeviation * 2
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp("cpu_fill randomTruncatedNormal", &result) {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let std2x = standardDeviation.element * 2
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp("cpu_fill randomTruncatedNormal", &result) {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean.element
        }
    }
}
