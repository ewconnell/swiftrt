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
import SwiftRTCuda

//==============================================================================
// CudaQueue fill functions
extension CudaQueue
{
    //--------------------------------------------------------------------------
    @inlinable public func copy<S,E>(
        from x: Tensor<S,E>, 
        to out: inout Tensor<S,E>
    ) {
        guard useGpu else { cpu_copy(from: x, to: &out); return }
        diagnostic(.queueGpu, "copy(from:to:) on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            x.withTensor(using: self) { xData, x in
                srtCopy(xData, x, o, oDesc, stream)
            }
        }
        cpuFallback(status) { $0.copy(from: x, to: &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E: StorageElement>(
        _ out: inout Tensor<S,E>,
        with element: E.Value
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_fill(&out, with: element); return }
        diagnostic(.queueGpu, "fill(_:element:) on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            withUnsafePointer(to: element) {
                srtFill(o, oDesc, $0, stream)
            }
        }
        cpuFallback(status) { $0.fill(&out, with: element) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E,B>(
        _ out: inout Tensor<S,E>,
        with range: Range<B>
    ) where E: StorageElement, E.Value: Numeric,
            B: SignedInteger, B.Stride: SignedInteger
    {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_fill(&out, with: range); return }
        diagnostic(.queueGpu, "fill(_:range:) on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            srtFillRange(o, oDesc, Int(range.lowerBound), stream)
        }
        cpuFallback(status) { $0.fill(&out, with: range) }
    }

    //--------------------------------------------------------------------------
    @inlinable func eye<S,E: StorageElement>(
        _ out: inout Tensor<S,E>,
        offset: Int
    ) where E.Value: Numeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_eye(&out, offset: offset); return }
        diagnostic(.queueGpu, "eye(_:offset:) on \(name)", categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            srtEye(o, oDesc, offset, stream)
        }
        cpuFallback(status) { $0.eye(&out, offset: offset) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomUniform out: inout Tensor<S,E>,
        _ lower: E.Value,
        _ upper: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomUniform: &out, lower, upper, seed); return
        }
        diagnostic(.queueGpu, "fill(randomUniform) on \(name)", categories: .queueGpu)

        let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            withUnsafePointer(to: lower) { l in
                withUnsafePointer(to: upper) { u in
                    srtFillRandomUniform(o, oDesc, l, u, seed64, stream)
                }
            }
        }
        cpuFallback(status) { $0.fill(randomUniform: &out, lower, upper, seed) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomNormal: &out, mean, std, seed); return
        }
        diagnostic(.queueGpu, "fill(randomNormal) on \(name)", categories: .queueGpu)

        let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            withUnsafePointer(to: mean) { m in
                withUnsafePointer(to: std) { s in
                    srtFillRandomNormal(o, oDesc, m, s, seed64, stream)
                }
            }
        }
        cpuFallback(status) { $0.fill(randomNormal: &out, mean, std, seed) }
    }

    //--------------------------------------------------------------------------
    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomNormal: &out, mean, std, seed); return
        }
        diagnostic(.queueGpu, "fill(randomNormal) on \(name)",
                   categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            srtFillRandomNormalTensorArgs(
                o, oDesc, 
                mean.deviceRead(using: self), 
                std.deviceRead(using: self),
                UInt64(msb: seed.op, lsb: seed.graph), 
                stream)
        }
        cpuFallback(status) { $0.fill(randomNormal: &out, mean, std, seed) }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: E.Value,
        _ std: E.Value,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &out, mean, std, seed)
            return
        }
        diagnostic(.queueGpu, "fill(randomTruncatedNormal) on \(name)", 
                   categories: .queueGpu)

        let seed64 = UInt64(msb: seed.op, lsb: seed.graph)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            withUnsafePointer(to: mean) { m in
                withUnsafePointer(to: std) { s in
                    srtFillRandomTruncatedNormal(o, oDesc, m, s, seed64, stream)
                }
            }
        }

        cpuFallback(status) {
            $0.fill(randomTruncatedNormal: &out, mean, std, seed)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal out: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ std: Tensor<S,E>,
        _ seed: RandomSeed
    ) where E.Value: BinaryFloatingPoint { 
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else {
            cpu_fill(randomTruncatedNormal: &out, mean, std, seed) 
            return
        }
        diagnostic(.queueGpu, "fill(randomTruncatedNormal) on \(name)", 
                   categories: .queueGpu)

        let status = out.withMutableTensor(using: self) { o, oDesc in
            srtFillRandomTruncatedNormalTensorArgs(
                o, oDesc, 
                mean.deviceRead(using: self), 
                std.deviceRead(using: self),
                UInt64(msb: seed.op, lsb: seed.graph), 
                stream)
        }

        cpuFallback(status) {
            $0.fill(randomTruncatedNormal: &out, mean, std, seed) 
        }
    }
}
