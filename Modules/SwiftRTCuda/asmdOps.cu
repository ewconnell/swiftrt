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
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <stdio.h>
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
    __device__ __forceinline__ E operator()(const E& a, const E& b) { return a + b; }
};

template<typename E>
struct Sub {
    __device__ __forceinline__ E operator()(const E& a, const E& b) { return a - b; }
};

template<typename E>
struct Mul {
    __device__ __forceinline__ E operator()(const E& a, const E& b) { return a * b; }
};

template<typename E>
struct Div {
    __device__ __forceinline__ E operator()(const E& a, const E& b) { return a / b; }
};

//==============================================================================
// kernels
//==============================================================================

/// abSingleSingle
// single op single --> flat
template<template<typename U> class Op, typename E>
__global__ void abSingleSingle(const E *a, const E *b, E *out, unsigned count) 
{
    Op<E> op;
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    out[i] = op(a[0], b[0]);
}

/// abFlatSingle
// flat op single --> flat
template<template<typename U> class Op, typename E>
__global__ void abFlatSingle(const E *a, const E *b, E *out, unsigned count) 
{
    Op<E> op;
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    out[i] = op(a[i], b[0]);
}

/// abSingleFlat
// single op flat --> flat
template<template<typename U> class Op, typename E>
__global__ void abSingleFlat(const E *a, const E *b, E *out, unsigned count) 
{
    Op<E> op;
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    out[i] = op(a[0], b[i]);
}

/// abFlatFlat
// flat op flat --> flat
template<template<typename U> class Op, typename E>
__global__ void abFlatFlat(const E *a, const E *b, E *out, unsigned count) 
{
    Op<E> op;
    unsigned i = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    out[i] = op(a[i], b[i]);
}

/// abStridedSingle
// strided op single --> flat
template<template<typename U> class Op, typename E, typename IndexA>
__global__ void abStridedSingle(
    const E *a, IndexA indexA, 
    const E *b, 
    E *out, 
    unsigned outEnd
) {
    Op<E> op;
    unsigned iout = Flat::linearIndex(blockIdx, blockDim, threadIdx);
    unsigned ia = indexA.linearIndex(blockIdx, blockDim, threadIdx);
    if (iout < outEnd) out[iout] = op(a[ia], b[0]);
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// mapStridedSingle
/// invokes the correct kernel to mapAB the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<template<typename U> class Op, typename E>
static cudaError_t mapStridedSingle(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    unsigned shiftCount = 0 
) {
    switch (aDesc.rank) {
    case 1: {
        unsigned count = oDesc.count;
        unsigned blocks = BLOCK_COUNT(count);
        unsigned threads = THREADS_PER_BLOCK;
        abStridedSingle<Op,E,Strided<1>> <<<blocks, threads, 0, stream>>>(a, Strided<1>(aDesc), b, out, count);
        break;
    }

    default: return cudaErrorNotSupported;
    }
    return cudaSuccess;
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
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);
    const E* b = static_cast<const E*>(pB);

    // the count is divided in cases where values are handled as short vectors
    unsigned count = shiftDownRoundingUp(oDesc.count, shiftCount);

    // make sure total count fits within Cuda limitations
    assert(count <= UINT32_MAX);
    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;
    
    if (bDesc.isSingle()) {
        if (aDesc.isSingle()) {
            // single op single --> flat
            abSingleSingle<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // flat op single --> flat
            abFlatSingle<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else {
            // strided op single --> flat
            return mapStridedSingle<Op,E>(a, aDesc, b, bDesc, out, oDesc, stream, shiftCount);
        }
    } else if (bDesc.isDense()) {
        if (aDesc.isSingle()) {
            // single op dense --> flat
            abSingleFlat<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // dense op dense --> flat
            abFlatFlat<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else {
            // strided op dense --> flat
        }
    } else {
        if (aDesc.isSingle()) {
            // single op strided --> flat
        } else if (aDesc.isDense()) {
            // dense op strided --> flat
        } else {
            // strided op strided --> flat
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
