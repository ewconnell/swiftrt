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
#include "fill_api.h"
#include "tensor.cuh"
#include "srt_traits.h"
#include "iterators.cuh"
#include "tensor_api.h"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//==============================================================================
// srtFill

// kernel
template <typename IterOut, typename E>
__global__ void mapFillWithElement(IterOut iterOut, E element) {
    auto p = typename IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) iterOut[p] = element;
}

template <typename O>
static cudaError_t fillWithElement(
    void *pOut,
    const void *pElement,
    size_t count,
    cudaStream_t stream
) {
    typedef typename packed<O>::type Out;
    Out *out = static_cast<Out*>(pOut);
    Out element = packed<O>::value(*static_cast<const O*>(pElement));

    auto iterO = Flat(out, count);

    // get tile and grid size for launch
    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    mapFillWithElement<<<grid, tile, 0, stream>>>(iterO, element);
    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// srtFillFlat
/// Fills the output buffer with the element value
///
/// - Parameters:
///  - type: the data type
///  - out: pointer to output buffer
///  - element: pointer to element fill value in host memory
///  - count: the number of elements to fill
///  - stream: the execution stream
cudaError_t srtFillFlat(
    srtDataType type,
    void* out,
    const void* element,
    size_t count,
    cudaStream_t stream
) {
    switch(type) {
        case real32F:  return fillWithElement<float>(out, element, count, stream);
        case real64F:  return fillWithElement<double>(out, element, count, stream);
        case real16F:  return fillWithElement<float16>(out, element, count, stream);
        case real16BF: return fillWithElement<bfloat16>(out, element, count, stream);
        case real16I:  return fillWithElement<int16_t>(out, element, count, stream);
        case real16U:  return fillWithElement<uint16_t>(out, element, count, stream);
        case real8I:   return fillWithElement<int8_t>(out, element, count, stream);
        case real8U:   return fillWithElement<uint8_t>(out, element, count, stream);
        case complex16F: return fillWithElement<Complex<float16> >(out, element, count, stream);
        case complex32F: return fillWithElement<Complex<float> >(out, element, count, stream);
        default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// srtFillRange

// kernel
// TODO: remove float cast. It currently is to get around missing bfloat cast
template <typename IterOut, typename E>
__global__ void mapFillRange(
    IterOut iterOut,
    const E first,
    const E last,
    const E step
) {
    auto lastPos = iterOut.count - 1;
    auto p = typename IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) {
        auto seqPos = iterOut.sequence(p);
        E value = first + (E(float(seqPos)) * step);
        iterOut[p] = seqPos == lastPos ? last : value; 
    }
}

template <typename E>
static cudaError_t fillRange(
    void *pOut,
    const void* pfirst,
    const void* plast,
    const void* pstep,
    const uint32_t count,
    cudaStream_t stream
) {
    E *out = static_cast<E *>(pOut);
    auto iterO = Flat(out, count);
    E first = *static_cast<const E*>(pfirst);
    E last  = *static_cast<const E*>(plast);
    E step  = *static_cast<const E*>(pstep);

    // get tile and grid size for launch
    dim3 tile = tileSize(count);
    dim3 grid = gridSize(count, tile);

    mapFillRange<<<grid, tile, 0, stream>>>(iterO, first, last, step);
    return cudaSuccess;
}

template <template<typename P, int R> class IterO, int Rank, typename E>
static cudaError_t fillRange(
    void *pOut, const TensorDescriptor &oDesc,
    const void* pfirst,
    const void* plast,
    const void* pstep,
    cudaStream_t stream
) {
    E *out = static_cast<E *>(pOut);
    auto iterO = IterO<E*, Rank>(out, oDesc);
    E first = *static_cast<const E*>(pfirst);
    E last  = *static_cast<const E*>(plast);
    E step  = *static_cast<const E*>(pstep);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    mapFillRange<<<grid, tile, 0, stream>>>(iterO, first, last, step);
    return cudaSuccess;
}

template <typename E, int R>
static cudaError_t selectFillRangeIndex(
    void *out, const TensorDescriptor &oDesc,
    const void* first,
    const void* last,
    const void* step,
    cudaStream_t stream
) {
    switch (oDesc.order) {
    case CUBLASLT_ORDER_ROW: return fillRange<E>(out, first, last, step, oDesc.count, stream);
    case CUBLASLT_ORDER_COL: return fillRange<StridedSeq,R,E>(out, oDesc, first, last, step, stream);
    default: return cudaErrorNotSupported;
    }
}

template <typename E>
static cudaError_t selectFillRangeRank(
    void *out, const TensorDescriptor &oDesc,
    const void* first,
    const void* last,
    const void* step,
    cudaStream_t stream
) {
    switch (oDesc.rank) {
    case 1: return selectFillRangeIndex<E,1>(out, oDesc, first, last, step, stream);
    case 2: return selectFillRangeIndex<E,2>(out, oDesc, first, last, step, stream);
    case 3: return selectFillRangeIndex<E,3>(out, oDesc, first, last, step, stream);
    default: return cudaErrorNotSupported;
    }
}

//------------------------------------------------------------------------------
/// srtFillRangeFlat
/// Fills the output with logical position indexes  
cudaError_t srtFillRangeFlat(
    srtDataType type,
    void* out,
    const void* first,
    const void* last,
    const void* step,
    size_t count,
    cudaStream_t stream
) {
    switch (type) {
    case real32F:  return fillRange<float>(out, first, last, step, count, stream);
    case real64F:  return fillRange<double>(out, first, last, step, count, stream);
    case real32I:  return fillRange<int32_t>(out, first, last, step, count, stream);

    // can't pack these types because the range is generated sequentially
    case real16F:  return fillRange<float16>(out, first, last, step, count, stream);
    case real16BF: return fillRange<bfloat16>(out, first, last, step, count, stream);
    case real32U:  return fillRange<uint32_t>(out, first, last, step, count, stream);
    case real16I:  return fillRange<int16_t>(out, first, last, step, count, stream);
    case real16U:  return fillRange<uint16_t>(out, first, last, step, count, stream);
    case real8I:   return fillRange<int8_t>(out, first, last, step, count, stream);
    case real8U:   return fillRange<uint8_t>(out, first, last, step, count, stream);

    case complex16F: return fillRange<Complex<float16> >(out, first, last, step, count, stream);
    case complex16BF: return fillRange<Complex<bfloat16> >(out, first, last, step, count, stream);
    case complex32F: return fillRange<Complex<float> >(out, first, last, step, count, stream);
    default: return cudaErrorNotSupported;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
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
    case real16F:  return selectFillRangeRank<float16>(out, oDesc, first, last, step, stream);
    case real16BF: return selectFillRangeRank<bfloat16>(out, oDesc, first, last, step, stream);
    case real64F:  return selectFillRangeRank<double>(out, oDesc, first, last, step, stream);
    case real32I:  return selectFillRangeRank<int32_t>(out, oDesc, first, last, step, stream);
    case real32U:  return selectFillRangeRank<uint32_t>(out, oDesc, first, last, step, stream);
    case real16I:  return selectFillRangeRank<int16_t>(out, oDesc, first, last, step, stream);
    case real16U:  return selectFillRangeRank<uint16_t>(out, oDesc, first, last, step, stream);
    case real8I:   return selectFillRangeRank<int8_t>(out, oDesc, first, last, step, stream);
    case real8U:   return selectFillRangeRank<uint8_t>(out, oDesc, first, last, step, stream);
    case complex16F: return selectFillRangeRank<Complex<float16> >(out, oDesc, first, last, step, stream);
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
