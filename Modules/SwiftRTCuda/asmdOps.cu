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
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Element wise Add, Subtract, Multiply, Divide ops
#include "asmdOps.h"
#include "kernelHelpers.h"
#include "index.h"

//==============================================================================
// ops
//==============================================================================

template<typename E>
struct Add {
    __device__ inline static E op(const E& a, const E& b) { return a + b; }
};

template<typename E>
struct Sub {
    __device__ inline static E op(const E& a, const E& b) { return a - b; }
};

template<typename E>
struct Mul {
    __device__ inline static E op(const E& a, const E& b) { return a * b; }
};

template<typename E>
struct Div {
    __device__ inline static E op(const E& a, const E& b) { return a / b; }
};

//==============================================================================
// kernels
//==============================================================================

//--------------------------------------
/// abSingleSingle
// single op single --> flat
template<typename F, typename E>
__global__ void abSingleSingle(const E *a, const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[0], b[0]);
}

//--------------------------------------
/// abFlatSingle
// flat op single --> flat
template<typename F, typename E>
__global__ void abFlatSingle(const E *a, const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[i], b[0]);
}

//--------------------------------------
/// abSingleFlat
// single op flat --> flat
template<typename F, typename E>
__global__ void abSingleFlat(const E *a, const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[0], b[i]);
}

//--------------------------------------
/// abFlatFlat
// flat op flat --> flat
template<typename F, typename E>
__global__ void abFlatFlat(const E *a, const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[i], b[i]);
}

//--------------------------------------
/// abStridedSingle
// strided op single --> flat
template<typename F, typename E, typename IndexA>
__global__ void abStridedSingle(const E *a, IndexA indexA, 
                                const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ia = indexA.linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[ia], b[0]);
}

//--------------------------------------
/// abSingleStrided
// single op strided --> flat
template<typename F, typename E, typename IndexB>
__global__ void abSingleStrided(const E *a, const E *b, 
                                IndexB indexB, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ib = indexB.linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[0], b[ib]);
}

//--------------------------------------
/// abStridedFlat
// strided op flat --> flat
template<typename F, typename E, typename IndexA>
__global__ void abStridedFlat(const E *a, IndexA indexA, 
                              const E *b, E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ia = indexA.linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[ia], b[i]);
}

//--------------------------------------
/// abFlatStrided
// strided op flat --> flat
template<typename F, typename E, typename IndexB>
__global__ void abFlatStrided(const E *a, const E *b,
                              IndexB indexB,  E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ib = indexB.linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[i], b[ib]);
}

//--------------------------------------
/// abStridedStrided
// strided op flat --> flat
template<typename F, typename E, typename IndexA, typename IndexB>
__global__ void abStridedStrided(const E *a, IndexA indexA, 
                                 const E *b, IndexB indexB,
                                 E *out, unsigned end) 
{
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ia = indexA.linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ib = indexB.linearIndex(blockIdx, blockDim, threadIdx);
    if (i < end) out[i] = F::op(a[ia], b[ib]);
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// mapStridedSingle
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<typename F, typename E, unsigned Rank>
static inline void mapStridedSingle(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream 
) {
    dim3 tile = tileSize<Rank>(oDesc);
    dim3 grid = gridSize<Rank>(oDesc, tile);
    abStridedSingle<F,E,Strided<Rank>> <<<grid, tile, 0, stream>>>(
        a, Strided<Rank>(aDesc), b, out, oDesc.spanCount);
}

//------------------------------------------------------------------------------
/// mapSingleStrided
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<typename F, typename E, unsigned Rank>
static inline void mapSingleStrided(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream 
) {
    dim3 tile = tileSize<Rank>(oDesc);
    dim3 grid = gridSize<Rank>(oDesc, tile);
    abSingleStrided<F,E,Strided<Rank>> <<<grid, tile, 0, stream>>>(
        a, b, Strided<Rank>(bDesc), out, oDesc.spanCount);
}

//------------------------------------------------------------------------------
/// mapStridedSingle
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<typename F, typename E, unsigned Rank>
static inline void mapStridedFlat(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream 
) {
    dim3 tile = tileSize<Rank>(oDesc);
    dim3 grid = gridSize<Rank>(oDesc, tile);
    abStridedFlat<F,E,Strided<Rank>> <<<grid, tile, 0, stream>>>(
        a, Strided<Rank>(aDesc), b, out, oDesc.spanCount);
}

//------------------------------------------------------------------------------
/// mapFlatStrided
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<typename F, typename E, unsigned Rank>
static inline void mapFlatStrided(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream 
) {
    dim3 tile = tileSize<Rank>(oDesc);
    dim3 grid = gridSize<Rank>(oDesc, tile);
    abFlatStrided<F,E,Strided<Rank>> <<<grid, tile, 0, stream>>>(
        a, b, Strided<Rank>(bDesc), out, oDesc.spanCount);
}

//------------------------------------------------------------------------------
/// mapStridedSingle
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<typename F, typename E, unsigned Rank>
static inline void mapStridedStrided(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream 
) {
    dim3 tile = tileSize<Rank>(oDesc);
    dim3 grid = gridSize<Rank>(oDesc, tile);
    abStridedStrided<F,E,Strided<Rank>, Strided<Rank>> <<<grid, tile, 0, stream>>>(
        a, Strided<Rank>(aDesc),
        b, Strided<Rank>(bDesc), 
        out, oDesc.spanCount);
}

//------------------------------------------------------------------------------
/// mapAB
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<template<typename U> class Op, typename E>
static cudaError_t mapAB(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    unsigned shiftCount = 0 
) {
    typedef Op<E> F;
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);
    const E* b = static_cast<const E*>(pB);

    // the count is divided in cases where values are handled as short vectors
    unsigned count = shiftDownRoundingUp(oDesc.count, shiftCount);

    // make sure total count fits within Cuda limitations
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);
    
    if (bDesc.isSingle()) {
        if (aDesc.isSingle()) {
            // single op single --> flat
            abSingleSingle<F,E><<<grid, tile, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // flat op single --> flat
            abFlatSingle<F,E><<<grid, tile, 0, stream>>>(a, b, out, count);

        } else {
            // strided op single --> flat
            switch (oDesc.rank) {
            case 1: mapStridedSingle<F,E,1>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: mapStridedSingle<F,E,2>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: mapStridedSingle<F,E,3>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else if (bDesc.isDense()) {
        if (aDesc.isSingle()) {
            // single op dense --> flat
            abSingleFlat<F,E><<<grid, tile, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // dense op dense --> flat
            abFlatFlat<F,E><<<grid, tile, 0, stream>>>(a, b, out, count);

        } else {
            // strided op flat --> flat
            switch (oDesc.rank) {
            case 1: mapStridedFlat<F,E,1>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: mapStridedFlat<F,E,2>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: mapStridedFlat<F,E,3>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        if (aDesc.isSingle()) {
            // single op strided --> flat
            switch (oDesc.rank) {
            case 1: mapSingleStrided<F,E,1>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: mapSingleStrided<F,E,2>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: mapSingleStrided<F,E,3>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        } else if (aDesc.isDense()) {
            // flat op strided --> flat
            switch (oDesc.rank) {
            case 1: mapFlatStrided<F,E,1>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: mapFlatStrided<F,E,2>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: mapFlatStrided<F,E,3>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        } else {
            // strided op strided --> flat
            switch (oDesc.rank) {
            case 1: mapStridedStrided<F,E,1>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: mapStridedStrided<F,E,2>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: mapStridedStrided<F,E,3>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaSuccess;
}

//------------------------------------------------------------------------------
// selectType
// converts from dynamic to static type and delegates for stride selection
template<template<typename U> class Op>
static cudaError_t selectType(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pbDesc);

    // must be same data type and rank, and output is dense
    assert(oDesc.isDense());
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type);
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == bDesc.order && aDesc.order == oDesc.order);
    
    switch(oDesc.type) {
        case CUDA_R_32F:  return mapAB<Op, float>(a, aDesc, b, bDesc, out, oDesc, stream);
        // case CUDA_R_16BF: return mapAB<Op, __nv_bfloat162>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case CUDA_R_16F:  return mapAB<Op, __half>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case CUDA_R_8I:   return mapAB<Op, int8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_8U:   return mapAB<Op, uint8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16I:  return mapAB<Op, int16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16U:  return mapAB<Op, uint16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_64F:  return mapAB<Op, double>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Add>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Sub>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Mul>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Div>(a, aDesc, b, bDesc, out, oDesc, stream);
}