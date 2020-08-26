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
#include <__clang_cuda_runtime_wrapper.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fillOps.h"
#include "dispatchHelpers.h"

//==============================================================================
// kernels
//==============================================================================

template<typename E, typename IndexO>
__global__ void mapFill(E *out, IndexO indexO, E element) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int i = indexO.linear(position);
        out[i] = element;
    }
}

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

//==============================================================================
// srtFill
template<typename E>
static cudaError_t fill(
    void* pOut, const TensorDescriptor& oDesc,
    const void* pElement,
    cudaStream_t stream
) {
    E* out = static_cast<E*>(pOut);
    E element = *static_cast<const E*>(pElement);
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);
    mapFill<E,Flat><<<grid, tile, 0, stream>>>(out, Flat(oDesc), element);
    return cudaSuccess;
}

cudaError_t srtFill(
    void* out, const srtTensorDescriptor* poDesc,
    const void* element,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);
    assert(oDesc.isDense());
    
    switch(oDesc.type) {
        case CUDA_R_32F:  return fill<float>(out, oDesc, element, stream);
        case CUDA_R_16BF: return fill<__nv_bfloat16>(out, oDesc, element, stream);
        case CUDA_R_16F:  return fill<__half>(out, oDesc, element, stream);
        case CUDA_R_8I:   return fill<int8_t>(out, oDesc, element, stream);
        case CUDA_R_8U:   return fill<uint8_t>(out, oDesc, element, stream);
        case CUDA_R_16I:  return fill<int16_t>(out, oDesc, element, stream);
        case CUDA_R_16U:  return fill<uint16_t>(out, oDesc, element, stream);
        case CUDA_R_64F:  return fill<double>(out, oDesc, element, stream);
        default: return cudaErrorNotSupported;
    }
}

//==============================================================================
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
    const uint64_t seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomNormal(
    void* out, const srtTensorDescriptor* oDesc,
    const void* mean,
    const void* std,
    const uint64_t seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomNormalTensorArgs(
    void* out, const srtTensorDescriptor* oDesc,
    const void* meanTensor,
    const void* stdTensor,
    const uint64_t seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomTruncatedNormal(
    void* out, const srtTensorDescriptor* oDesc,
    const void* mean,
    const void* std,
    const uint64_t seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtFillRandomTruncatedNormalTensorArgs(
    void* out, const srtTensorDescriptor* oDesc,
    const void* meanTensor,
    const void* stdTensor,
    const uint64_t seed,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
