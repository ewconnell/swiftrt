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
extension CpuQueue
{
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E: StorageElement>(
        _ x: inout Tensor<S,E>,
        with element: E.Value
    ) where S: TensorShape { cpu_fill(&x, with: element) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        _ x: inout Tensor<S,E>,
        with range: Range<Int>
    ) where E: StorageElement, E.Value: Numeric {
        cpu_fill(&x,
                 from: E.Value(exactly: range.lowerBound)!,
                 to: E.Value(exactly: range.upperBound - 1)!,
                 by: E.Value(exactly: 1)!)
    }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        _ x: inout Tensor<S,E>,
        from first: E.Value,
        to last: E.Value,
        by step: E.Value
    ) where E: StorageElement, E.Value: Numeric {
        cpu_fill(&x, from: first, to: last, by: step)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ x: inout Tensor<S,E>,
        offset: Int
    ) where S: TensorShape, E.Value: Numeric { cpu_eye(&x, offset: offset) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform x: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomUniform: &x, lower, upper, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal x: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomNormal: &x, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal x: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomNormal: &x, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal x: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomTruncatedNormal: &x, mean, standardDeviation, seed) }
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal x: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint
    { cpu_fill(randomTruncatedNormal: &x, mean, standardDeviation, seed) }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable func cpu_eye<S,E>(
        _ result: inout Tensor<S,E>,
        offset: Int
    ) where S: TensorShape, E.Value: Numeric {
        mapOp(&result, "eye()") { 0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        _ result: inout Tensor<S,E>,
        with element: E.Value
    ) where S: TensorShape {
        mapOp(&result, "fill(with: \(element)") { element }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        _ result: inout Tensor<S,E>,
        from first: E.Value,
        to last: E.Value,
        by step: E.Value
    ) where E.Value: Numeric {
        mapOp(from: first, to: last, by: step, &result,
              "fill(from: \(first), to: \(last), by: \(step))")
    }
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint {
        let scale = Double(upper - lower) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp(&result, "fill(randomUniform: lower: \(lower), upper: \(upper), seed: \(seed)") {
            E.Value(Double(generator.next()) * scale) + lower
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint {
        let scale = Double(std) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp(&result, "fill(randomNormal: mean: \(mean), std: \(std), seed: \(seed)") {
            E.Value(Double(generator.next()) * scale) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func cpu_fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint {
        assert(std.count == 1 && mean.count == 1)
        let scale = Double(std.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp(&result, "fill(randomNormal: mean: \(mean.name), std: \(std.name), seed: \(seed)") {
            E.Value(Double(generator.next()) * scale) + mean.element
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint {
        let std2x = std * 2
        let scale = Double(std) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp(&result, "fill(randomTruncatedNormal: mean: \(mean), std: \(std), seed: \(seed)") {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E.Value: BinaryFloatingPoint {
        assert(std.count == 1 && mean.count == 1)
        let std2x = std.element * 2
        let scale = Double(std.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)
        mapOp(&result, "fill(randomTruncatedNormal: mean: \(mean.name), std: \(std.name), seed: \(seed)") {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean.element
        }
    }
}
