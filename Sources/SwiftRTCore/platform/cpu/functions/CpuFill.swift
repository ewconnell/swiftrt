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
extension CpuQueue {
    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        _ out: inout Tensor<S,E>,
        with element: E.Value
    ) {
        cpu_fill(&out, with: element)
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        _ out: inout Tensor<S,E>,
        from first: E.Value,
        to last: E.Value,
        by step: E.Value
    ) where E.Value: Numeric {
        cpu_fill(&out, from: first, to: last, by: step)
    }
    
    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ out: inout Tensor<S,E>,
        offset: Int
    ) where E.Value: Numeric {
        cpu_eye(&out, offset: offset)
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform out: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        cpu_fill(randomUniform: &out, lower, upper, seed)
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        cpu_fill(randomNormal: &out, mean, standardDeviation, seed)
    }

    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        cpu_fill(randomNormal: &out, mean, standardDeviation, seed)
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ standardDeviation: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        cpu_fill(randomTruncatedNormal: &out, mean, standardDeviation, seed)
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        cpu_fill(randomTruncatedNormal: &out, mean, standardDeviation, seed)
    }
}

//==============================================================================
// Cpu device queue function implementations
extension CpuFunctions where Self: DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable func cpu_eye<S,E>(
        _ out: inout Tensor<S,E>,
        offset: Int
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "eye(\(out.name)) on \(name)",
                   categories: .queueCpu)
        mapOp(&out) { 0 }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        _ out: inout Tensor<S,E>,
        with element: E.Value
    ) {
        diagnostic(.queueCpu, "fill(\(out.name), with: \(element)) on \(name)",
                   categories: .queueCpu)
        mapOp(&out) { element }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        _ out: inout Tensor<S,E>,
        from first: E.Value,
        to last: E.Value,
        by step: E.Value
    ) where E.Value: Numeric {
        diagnostic(.queueCpu, "fill(\(out.name), from: \(first), to: \(last)," +
                    " by: \(step)) on \(name)", categories: .queueCpu)
        mapOp(from: first, to: last, by: step, &out)
    }
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomUniform out: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        diagnostic(.queueCpu, "fill(randomUniform: \(out.name), lower: " +
                    "\(lower), upper: \(upper), seed: \(seed)) on \(name)",
                   categories: .queueCpu)
        let scale = Double(upper - lower) / Double(UInt64.max)
        var generator = Platform.createRandomNumberGenerator(using: seed)
        mapOp(&out) {
            E.Value(Double(generator.next()) * scale) + lower
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        diagnostic(.queueCpu, "fill(randomNormal: \(out.name), mean: " +
                    "\(mean), std: \(std), seed: \(seed)) on \(name)",
                   categories: .queueCpu)
        let scale = Double(std) / Double(UInt64.max)
        var generator = Platform.createRandomNumberGenerator(using: seed)
        mapOp(&out) {
            E.Value(Double(generator.next()) * scale) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func cpu_fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        assert(std.count == 1 && mean.count == 1)
        diagnostic(.queueCpu, "fill(randomNormal: \(out.name), mean: " +
                    "\(mean.name), std: \(std.name), seed: \(seed)) on \(name)",
                   categories: .queueCpu)
        let scale = Double(std.element) / Double(UInt64.max)
        var generator = Platform.createRandomNumberGenerator(using: seed)
        mapOp(&out) {
            E.Value(Double(generator.next()) * scale) + mean.element
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        diagnostic(.queueCpu, "fill(randomTruncatedNormal: \(out.name), mean: " +
                    "\(mean), std: \(std), seed: \(seed)) on \(name)",
                   categories: .queueCpu)
        let std2x = std * 2
        let scale = Double(std) / Double(UInt64.max)
        var generator = Platform.createRandomNumberGenerator(using: seed)
        mapOp(&out) {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func cpu_fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        assert(std.count == 1 && mean.count == 1)
        diagnostic(.queueCpu, "fill(randomTruncatedNormal: \(out.name), " +
                    "mean: \(mean.name), std: \(std.name), seed: \(seed)) on \(name)",
                   categories: .queueCpu)
        let std2x = std.element * 2
        let scale = Double(std.element) / Double(UInt64.max)
        var generator = Platform.createRandomNumberGenerator(using: seed)
        mapOp(&out) {
            let a = Double(generator.next()) * scale
            return E.Value(a).clamped(to: -std2x...std2x) + mean.element
        }
    }
}
