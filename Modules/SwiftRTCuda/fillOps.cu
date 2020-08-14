//******************************************************************************
// Copyright 2019 Google LLC
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
#include <assert.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fillOps.h"

//==============================================================================
// kernels
//==============================================================================

//==============================================================================
// dynamic dispatch functions
//==============================================================================


//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtCopy(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFill(
    void* out, const srtTensorDescriptor* oDesc,
    const void* element,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillWithRange(
    void* out, const srtTensorDescriptor* oDesc,
    const long lower,
    const long upper, 
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtEye(
    void* out, const srtTensorDescriptor* oDesc,
    const long offset,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomUniform(
    void* out, const srtTensorDescriptor* oDesc,
    const void* lower,
    const void* upper,
    const uint64_t* seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomNormal(
    void* out, const srtTensorDescriptor* oDesc,
    const void* mean,
    const void* std,
    const uint64_t* seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomNormalTensorArgs(
    void* out, const srtTensorDescriptor* oDesc,
    void* meanTensor,
    void* stdTensor,
    const uint64_t* seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomTruncatedNormal(
    void* out, const srtTensorDescriptor* oDesc,
    const void* mean,
    const void* std,
    const uint64_t* seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomTruncatedNormalTensorArgs(
    void* out, const srtTensorDescriptor* oDesc,
    void* meanTensor,
    void* stdTensor,
    const uint64_t* seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
