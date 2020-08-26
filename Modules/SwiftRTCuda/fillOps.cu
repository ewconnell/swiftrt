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
#include <bits/stdint-uintn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fillOps.h"
#include "dispatchHelpers.h"

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

// kernel
template <typename E, typename IndexO>
__global__ void mapElementFill(E *out, IndexO indexO, E element) {
  auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
  if (indexO.isInBounds(position)) {
    int i = indexO.linear(position);
    out[i] = element;
  }
}

template <typename E>
static cudaError_t elementFill(void *pOut, const TensorDescriptor &oDesc,
                               const E element, cudaStream_t stream,
                               const int shiftCount = 0) {
  E *out = static_cast<E *>(pOut);
  int count = shiftDownRoundingUp(oDesc.count, shiftCount);
  dim3 tile = tileSize(count);
  dim3 grid = gridSize<1>(oDesc, tile);
  mapElementFill<E, Flat><<<grid, tile, 0, stream>>>(out, Flat(oDesc), element);
  return cudaSuccess;
}

//------------------------------------------------------------------------------
/// srtFill
/// Fills the output buffer with the element value
///
/// - Parameters:
///  - out: pointer to device memory output buffer
///  - poDesc: pointer to output srtTensorDescriptor
///  - element: pointer to element fill value in host memory
///  - stream: the execution stream
cudaError_t srtFill(
    void* out, const srtTensorDescriptor* poDesc,
    const void* element,
    cudaStream_t stream
) {
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);
    assert(oDesc.isDense());

    // The output type is converted to a packed type if possible for a faster
    // fill operation. The output buffer space is guaranteed to be rounded
    // up to a multiple of the largest packed type so we don't have to worry
    // about writing out of bounds.
    switch(oDesc.type) {
        case CUDA_R_32F:
          return elementFill<float>(out, oDesc, *(float *)element, stream);
        case CUDA_R_64F:
          return elementFill<double>(out, oDesc, *(double *)element, stream);

        case CUDA_R_16F: 
        case CUDA_R_16BF:
        case CUDA_R_16I:
        case CUDA_R_16U:
            // pack 16 bit elements
            return elementFill<uint32_t>(out, oDesc, fillWord<uint16_t>(element), stream, 1);

        case CUDA_R_8I: 
        case CUDA_R_8U:
            // pack 8 bit elements
            return elementFill<uint32_t>(out, oDesc, fillWord<uint8_t>(element), stream, 2);
        default: return cudaErrorNotSupported;
    }
}

//==============================================================================
/// srtFillRange
/// Fills the output with logical position indexes  
cudaError_t srtFillRange(
    void* out, const srtTensorDescriptor* oDesc,
    const long lower,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

//==============================================================================
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
