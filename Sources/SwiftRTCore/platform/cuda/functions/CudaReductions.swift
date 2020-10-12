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
import Numerics

//==============================================================================
// CudaQueue functions
extension CudaQueue {
    //--------------------------------------------------------------------------
    @inlinable public func abssum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: SignedNumeric & Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_abssum(x, &out); return }
        diagnostic(.queueGpu, "abssum(\(x.name)) on \(name)",
            categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtAbsSum(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }        
        cpuFallback(status) { $0.abssum(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func all<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_all(x, &out); return }
        diagnostic(.queueGpu, "all(\(x.name)) on \(name)",
            categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtAll(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }        
        cpuFallback(status) { $0.all(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func any<S>(
        _ x: Tensor<S,Bool>,
        _ out: inout Tensor<S,Bool>
    ) {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_any(x, &out); return }
        diagnostic(.queueGpu, "any(\(x.name)) on \(name)",
            categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtAny(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }        
        cpuFallback(status) { $0.any(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func sum<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AdditiveArithmetic {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_sum(x, &out); return }
        diagnostic(.queueGpu, "sum(\(x.name)) on \(name)", categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtSum(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }
        cpuFallback(status) { $0.sum(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func mean<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: AlgebraicField {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_mean(x, &out); return }
        diagnostic(.queueGpu, "mean(\(x.name)) on \(name)",
            categories: .queueGpu)

        cpuFallback(cudaErrorNotSupported) { $0.mean(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func min<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_min(x, &out); return }
        diagnostic(.queueGpu, "min(\(x.name)) on \(name)",
            categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtMinValue(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }
        cpuFallback(status) { $0.min(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func argmin<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_argmin(x, &out); return }
        diagnostic(.queueGpu, "argmin(\(x.name)) on \(name)",
            categories: .queueGpu)

        cpuFallback(cudaErrorNotSupported) { $0.argmin(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func max<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_max(x, &out); return }
        diagnostic(.queueGpu, "max(\(x.name)) on \(name)",
            categories: .queueGpu)
        var status = cudaErrorNotSupported;

        if out.count == 1 {
            status = out.withMutableTensor(using: self) { o, oDesc in
                x.withTensor(using: self) { x, xDesc in
                    srtMaxValue(x, xDesc, o, oDesc, stream)
                }
            }
        } else {

        }
        cpuFallback(status) { $0.max(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func argmax<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Comparable {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_argmax(x, &out); return }
        diagnostic(.queueGpu, "argmax(\(x.name)) on \(name)",
            categories: .queueGpu)

        cpuFallback(cudaErrorNotSupported) { $0.argmax(x, &out) }
    }

    //--------------------------------------------------------------------------
    @inlinable public func prod<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_prod(x, &out); return }
        diagnostic(.queueGpu, "prod(\(x.name)) on \(name)",
            categories: .queueGpu)

        cpuFallback(cudaErrorNotSupported) { $0.prod(x, &out) }
    }
    //--------------------------------------------------------------------------
    @inlinable public func prodNonZeros<S,E>(
        _ x: Tensor<S,E>,
        _ out: inout Tensor<S,E>
    ) where E.Value: Numeric {
        assert(out.isContiguous, _messageElementsMustBeContiguous)
        guard useGpu else { cpu_prodNonZeros(x, &out); return }
        diagnostic(.queueGpu, "prodNonZeros(\(x.name)) on \(name)",
            categories: .queueGpu)

        cpuFallback(cudaErrorNotSupported) { $0.prodNonZeros(x, &out) }
    }
}
