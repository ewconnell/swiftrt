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
#include "fill_api.cuh"
// #include "srt_traits.cuh"
#include "float16.cuh"
#include "bfloat16.cuh"
#include "complex.cuh"
#include "iterators.cuh"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//==============================================================================
// srtCopy
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
template <typename IterOut, typename E>
__global__ void mapFillWithElement(IterOut iterOut, E element) {
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) iterOut[p] = element;
}

template <typename O>
static cudaError_t fillWithElement(
    void *pOut, const TensorDescriptor &oDesc,
    const void *pElement,
    cudaStream_t stream
) {
    typedef typename packed<O>::type Out;
    Out *out = static_cast<Out*>(pOut);
    Out element = packed<O>::value(*static_cast<const O*>(pElement));

    auto iterO = Flat(out, oDesc.count);

    // get tile and grid size for launch
    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    mapFillWithElement<<<grid, tile, 0, stream>>>(iterO, element);
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
        case real32F:  return fillWithElement<float>(out, oDesc, element, stream);
        case real64F:  return fillWithElement<double>(out, oDesc, element, stream);
        case real16F:  return fillWithElement<float16>(out, oDesc, element, stream);
        case real16BF: return fillWithElement<bfloat16>(out, oDesc, element, stream);
        case real16I:  return fillWithElement<int16_t>(out, oDesc, element, stream);
        case real16U:  return fillWithElement<uint16_t>(out, oDesc, element, stream);
        case real8I:   return fillWithElement<int8_t>(out, oDesc, element, stream);
        case real8U:   return fillWithElement<uint8_t>(out, oDesc, element, stream);
        case complex32F: return fillWithElement<Complex<float> >(out, oDesc, element, stream);
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
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) {
        auto seqPos = iterOut.sequence(p);
        E value = first + (E(float(seqPos)) * step);
        iterOut[p] = seqPos == lastPos ? last : value; 
    }
}

template <typename E>
static cudaError_t fillRange(
    void *pOut,
    const E first,
    const E last,
    const E step,
    const uint32_t count,
    cudaStream_t stream
) {
    E *out = static_cast<E *>(pOut);
    auto iterO = Flat(out, count);

    // get tile and grid size for launch
    dim3 tile = tileSize(count);
    dim3 grid = gridSize(count, tile);

    mapFillRange<<<grid, tile, 0, stream>>>(iterO, first, last, step);
    cudaStreamSynchronize(stream);
    return cudaSuccess;
}

template <template<typename P, int R> class IterO, int Rank, typename E>
static cudaError_t fillRange(
    void *pOut, const TensorDescriptor &oDesc,
    const E first,
    const E last,
    const E step,
    cudaStream_t stream
) {
    E *out = static_cast<E *>(pOut);
    auto iterO = IterO<E*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    mapFillRange<<<grid, tile, 0, stream>>>(iterO, first, last, step);
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
    case CUBLASLT_ORDER_ROW: return fillRange(out, first, last, step, oDesc.count, stream);
    case CUBLASLT_ORDER_COL: return fillRange<StridedSeq,R,E>(out, oDesc, first, last, step, stream);
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
    case real16F:  return selectFillRangeRank<float16>(out, oDesc, first, last, step, stream);
    case real16BF: return selectFillRangeRank<bfloat16>(out, oDesc, first, last, step, stream);
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
