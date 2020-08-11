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
// #if (__CUDA_ARCH__ < 800)
// __device__ __forceinline__ __nv_bfloat162 operator+(const __nv_bfloat162& l, const __nv_bfloat162& r) {
//     __nv_bfloat162 c;
//     c.x = __float2bfloat16_rn(__bfloat162float(l.x) + __bfloat162float(r.x));
//     c.y = __float2bfloat16_rn(__bfloat162float(l.y) + __bfloat162float(r.y));
//     return c;
// }
// #endif

__device__ inline __nv_bfloat162 add(const __nv_bfloat162& l, const __nv_bfloat162& r) {
    __nv_bfloat162 c;
    c.x = __float2bfloat16_rn(__bfloat162float(l.x) + __bfloat162float(r.x));
    c.y = __float2bfloat16_rn(__bfloat162float(l.y) + __bfloat162float(r.y));
    return c;
}

// template<typename E>
// __global__ void add_bfloat162(
//     const void *va, int strideA,
//     const void *vb, int strideB,
//     void *vc,
//     unsigned count
// ) {
//     auto a = static_cast<const E*>(va);
//     auto b = static_cast<const E*>(vb);
//     auto c = static_cast<E*>(vc);

//     GRID_STRIDE_LOOP(ai, strideA, bi, strideB, ci, count) {
//         #if (__CUDA_ARCH__ >= 800)
//             c[ci] = a[ai] + b[bi];
//         #else
//             c[ci] = add(a[ai], b[bi]);
//         #endif
//     }
// }

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
template<template<typename U> class Op, typename E>
__global__ void abSingleSingle(const E *a, const E *b, E *out, uint32_t count) 
{
    Op<E> op;
    E element = op(a[0], b[0]);
    GRID_LOOP(i, count) {
        out[i] = element;
    }
}

template<template<typename U> class Op, typename E>
__global__ void abFlatSingle(const E *a, const E *b, E *out, uint32_t count) 
{
    Op<E> op;
    GRID_LOOP(i, count) {
        out[i] = op(a[i], b[0]);
    }
}

template<template<typename U> class Op, typename E>
__global__ void abSingleFlat(const E *a, const E *b, E *out, uint32_t count) 
{
    Op<E> op;
    GRID_LOOP(i, count) {
        out[i] = op(a[0], b[i]);
    }
}

template<template<typename U> class Op, typename E>
__global__ void abFlatFlat(const E *a, const E *b, E *out, uint32_t count) 
{
    Op<E> op;
    GRID_LOOP(i, count) {
        out[i] = op(a[i], b[i]);
    }
}

//------------------------------------------------------------------------------
/// combine
/// invokes the correct kernel to combine the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<template<typename U> class Op, int R, typename E>
static void combine(
    const void* pA, const srtTensorDescriptor* paDesc,
    const void* pB, const srtTensorDescriptor* pbDesc,
    void* pOut, const srtTensorDescriptor* poDesc,
    cudaStream_t stream,
    unsigned shiftCount = 0 
) {
    // statically cast types from C interface to use with c++ templates
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);
    const E* b = static_cast<const E*>(pB);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pbDesc);

    // the count is divided in cases where values are handled as short vectors
    unsigned count = shiftDownRoundingUp(oDesc.count, shiftCount);

    // make sure total count fits within Cuda limitations
    assert(count <= UINT32_MAX);
    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;
    
    if (bDesc.isSingle()) {
        if (aDesc.isSingle()) {
            // single op single --> dense
            abSingleSingle<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // dense op single --> dense
            abFlatSingle<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else {
            // strided op single --> dense
        }
    } else if (bDesc.isDense()) {
        if (aDesc.isSingle()) {
            // single op dense --> dense
            abSingleFlat<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else if (aDesc.isDense()) {
            // dense op dense --> dense
            abFlatFlat<Op,E><<<blocks, threads, 0, stream>>>(a, b, out, count);

        } else {
            // strided op dense --> dense
        }
    } else {
        if (aDesc.isSingle()) {
            // single op strided --> dense
        } else if (aDesc.isDense()) {
            // dense op strided --> dense
        } else {
            // strided op strided --> dense
        }
    }
}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

void srtAddR1Float(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,1,float>(a, aDesc, b, bDesc, out, oDesc, stream);
}

void srtAddR2Float(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,2,float>(a, aDesc, b, bDesc, out, oDesc, stream);
}

void srtAddR3Float(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,3,float>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//==============================================================================

void srtAddR1Float16(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,1,__half>(a, aDesc, b, bDesc, out, oDesc, stream);
}

void srtAddR2Float16(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,2,__half>(a, aDesc, b, bDesc, out, oDesc, stream);
}

void srtAddR3Float16(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    combine<Add,3,__half>(a, aDesc, b, bDesc, out, oDesc, stream);
}
