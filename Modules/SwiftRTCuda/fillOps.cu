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

#include "commonCDefs.h"
#include "fillOps.h"
#include "dispatchHelpers.h"
#include "index.h"

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
static cudaError_t elementFill(
    void *pOut, const TensorDescriptor &oDesc,
    const E element, cudaStream_t stream,
    const int shiftCount = 0
) {
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
        case real32F:
          return elementFill<float>(out, oDesc, *(float *)element, stream);
        case real64F:
          return elementFill<double>(out, oDesc, *(double *)element, stream);

        case real16F: 
        case real16BF:
        case real16I:
        case real16U:
            // pack 16 bit elements
            return elementFill<uint32_t>(out, oDesc, fillWord<uint16_t>(element), stream, 1);

        case real8I: 
        case real8U:
            // pack 8 bit elements
            return elementFill<uint32_t>(out, oDesc, fillWord<uint8_t>(element), stream, 2);

        case complex32F:
            return elementFill<Complex<float> >(out, oDesc, *(Complex<float> *)element, stream);
        default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// srtFillRange

// kernel
// TODO: remove float cast. It currently is to get around missing bfloat cast
template <typename E, typename IndexO>
__global__ void mapFillRange(
    E *out, IndexO indexO,
    const E first,
    const E last,
    const E step
) {
    auto lastIndex = indexO.count - 1;
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        auto i = indexO.linear(position);
        auto seq = indexO.sequence(position);
        printf("p: (%d, %d), seq: %d, i: %d\n", position[0], position[1], seq, i);
        E value = first + (E(float(seq)) * step);
        out[i] = i == lastIndex ? last : value; 
    }
}

template <typename E, typename IndexO>
static cudaError_t fillRange(
    void *pOut, const TensorDescriptor &oDesc,
    const E first,
    const E last,
    const E step,
    cudaStream_t stream
) {
    E *out = static_cast<E *>(pOut);
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);
    mapFillRange<E,IndexO><<<grid, tile, 0, stream>>>(out, IndexO(oDesc), first, last, step);
    cudaStreamSynchronize(stream);
    return cudaSuccess;
}

template <typename E, int R>
static cudaError_t selectFillRangeIndex(
    void *out, const TensorDescriptor &oDesc,
    const E first,
    const E last,
    const E step,
    cudaStream_t stream
) {
    switch (oDesc.order) {
    case CUBLASLT_ORDER_ROW: return fillRange<E,Flat>(out, oDesc, first, last, step, stream);
    case CUBLASLT_ORDER_COL: return fillRange<E,LogicalStrided<R> >(out, oDesc, first, last, step, stream);
    default: return cudaErrorNotSupported;
    }
}

template <typename E>
static cudaError_t selectFillRangeRank(
    void *out, const TensorDescriptor &oDesc,
    const void* pfirst,
    const void* plast,
    const void* pstep,
    cudaStream_t stream
) {
    E first = *static_cast<const E*>(pfirst);
    E last  = *static_cast<const E*>(plast);
    E step  = *static_cast<const E*>(pstep);

    switch (oDesc.rank) {
    case 1: return selectFillRangeIndex<E,1>(out, oDesc, first, last, step, stream);
    case 2: return selectFillRangeIndex<E,2>(out, oDesc, first, last, step, stream);
    case 3: return selectFillRangeIndex<E,3>(out, oDesc, first, last, step, stream);
    default: return cudaErrorNotSupported;
    }
}

/// srtFillRange
/// Fills the output with logical position indexes  
cudaError_t srtFillRange(
    void* out, const srtTensorDescriptor* poDesc,
    const void* first,
    const void* last,
    const void* step,
    cudaStream_t stream
) {
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);
    assert(oDesc.isDense());

    switch (oDesc.type) {
    case real32F:  return selectFillRangeRank<float>(out, oDesc, first, last, step, stream);
    case real16F:  return selectFillRangeRank<__half>(out, oDesc, first, last, step, stream);
    case real16BF: return selectFillRangeRank<__nv_bfloat16>(out, oDesc, first, last, step, stream);
    case real64F:  return selectFillRangeRank<double>(out, oDesc, first, last, step, stream);
    case real32I:  return selectFillRangeRank<int32_t>(out, oDesc, first, last, step, stream);
    case real32U:  return selectFillRangeRank<uint32_t>(out, oDesc, first, last, step, stream);
    case real16I:  return selectFillRangeRank<int16_t>(out, oDesc, first, last, step, stream);
    case real16U:  return selectFillRangeRank<uint16_t>(out, oDesc, first, last, step, stream);
    case real8I:   return selectFillRangeRank<int8_t>(out, oDesc, first, last, step, stream);
    case real8U:   return selectFillRangeRank<uint8_t>(out, oDesc, first, last, step, stream);
    case complex32F: return selectFillRangeRank<Complex<float> >(out, oDesc, first, last, step, stream);
    default: return cudaErrorNotSupported;
    }
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
