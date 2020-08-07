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
#include <cstddef>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Element wise Add, Subtract, Multiply, Divide ops
#include "asmdOps.h"
#include "kernelHelpers.h"

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

// template<typename T>
// __global__ void add_bfloat162(
//     const void *va, int strideA,
//     const void *vb, int strideB,
//     void *vc,
//     unsigned count
// ) {
//     auto a = static_cast<const T*>(va);
//     auto b = static_cast<const T*>(vb);
//     auto c = static_cast<T*>(vc);

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

template<typename T>
struct Add {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return a + b; }
};

template<typename T>
struct Sub {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return a - b; }
};

template<typename T>
struct Mul {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return a * b; }
};

template<typename T>
struct Div {
    __device__ __forceinline__ T operator()(const T& a, const T& b) { return a / b; }
};

//==============================================================================
// buffer iterable kernels
//==============================================================================

//------------------------------------------------------------------------------
// tensor2
template<template<typename U> class Op, typename T>
__global__ void tensor2(const T *a, const T *b, T *out, unsigned count) {
    Op<T> op;
    GRID_LOOP(i, count) {
        out[i] =  op(a[i], b[i]);
    }
}

//------------------------------------------------------------------------------
// tensorScalar
template<template<typename U> class Op, typename T>
__global__ void tensorScalar(const T *elements, const T *scalar, T *out, unsigned count) {
    Op<T> op;
    GRID_LOOP(i, count) {
        out[i] = op(elements[i], scalar[0]);
    }
}

//------------------------------------------------------------------------------
// scalarTensor
template<template<typename U> class Op, typename T>
__global__ void scalarTensor(const T *scalar, const T *elements, T *out, unsigned count) {
    Op<T> op;
    GRID_LOOP(i, count) {
        out[i] = op(scalar[0], elements[i]);
    }
}

//------------------------------------------------------------------------------
/// combine
/// invokes the correct kernel to combine the elements of the two tensors
/// handling the cases of elements and single scalar sets.
///
template<template<typename U> class Op, typename T>
static void combine(
    unsigned blocks, 
    unsigned threads,
    const void *pA, size_t strideA, 
    const void *pB, size_t strideB,
    void *pOut, size_t count,
    cudaStream_t stream
) {
    const T* a = static_cast<const T*>(pA);
    const T* b = static_cast<const T*>(pB);
    T* out = static_cast<T*>(pOut);

    if (strideA == 1 && strideB == 1) {
        // combine two sets of elements
        tensor2<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count);

    } else if (strideA == 1 && strideB == 0) {
        // combine elements with a scalar
        tensorScalar<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count); 

    } else if (strideA == 0 && strideB == 1) {
        // combine a scalar with elements
        scalarTensor<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count); 
    }
}

//------------------------------------------------------------------------------
// resolveType
// this function is for dense tensors that can be flattened where
// `isBufferIterable == true`, so strides must equal 0 or 1
template<template<typename U> class Op>
static cudaError_t resolveType(
    cudaDataType_t type, 
    const void *a, size_t strideA, 
    const void *b, size_t strideB,
    void *out, size_t count,
    cudaStream_t stream
) {
    // make sure sizes fit within Cuda limitations
    assert(count <= UINT32_MAX);
    assert(strideA == 0 || strideA == 1 && strideB == 0 || strideB == 1);
    KernelPreCheck(stream);

    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;

    switch(type) {
        case CUDA_R_32F: combine<Op, float>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        // *** Figure out how to define operator+ for this type so it works in emulation mode
        // case CUDA_R_16BF: {
        //     unsigned n = shiftDownRoundingUp(countC, 1);
        //     addScalar_bfloat162<__nv_bfloat162><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, pScalar, c, n);
        //     break;
        // }
        case CUDA_R_16F: {
            unsigned n = shiftDownRoundingUp(count, 1);
            combine<Add, __half>(BLOCK_COUNT(n), threads, a, strideA, b, strideB, out, count, stream); break;
            break;
        }
        case CUDA_R_8I:  combine<Op, int8_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        case CUDA_R_8U:  combine<Op, uint8_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        case CUDA_R_16I: combine<Op, int16_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        case CUDA_R_16U: combine<Op, uint16_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        case CUDA_R_64F: combine<Op, double>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        default: printf("cudaDataType_t not implemented"); exit(1);
    }

    return KernelPostCheck(stream);
}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
// srtAdd
cudaError_t srtAdd(
    cudaDataType_t type, 
    const void *a, size_t strideA, 
    const void *b, size_t strideB,
    void *out, size_t count,
    cudaStream_t stream
) {
    return resolveType<Add>(type, a, strideA, b, strideB, out, count, stream);
}

//------------------------------------------------------------------------------
// srtSub
cudaError_t srtSub(
    cudaDataType_t type, 
    const void *a, size_t strideA, 
    const void *b, size_t strideB,
    void *out, size_t count,
    cudaStream_t stream
) {
    return resolveType<Sub>(type, a, strideA, b, strideB, out, count, stream);
}

//------------------------------------------------------------------------------
// srtMul
cudaError_t srtMul(
    cudaDataType_t type, 
    const void *a, size_t strideA, 
    const void *b, size_t strideB,
    void *out, size_t count,
    cudaStream_t stream
) {
    return resolveType<Mul>(type, a, strideA, b, strideB, out, count, stream);
}

//------------------------------------------------------------------------------
// srtDiv
cudaError_t srtDiv(
    cudaDataType_t type, 
    const void *a, size_t strideA, 
    const void *b, size_t strideB,
    void *out, size_t count,
    cudaStream_t stream
) {
    return resolveType<Div>(type, a, strideA, b, strideB, out, count, stream);
}

//******************************************************************************
//******************************************************************************

//==============================================================================
// strided index kernels
//==============================================================================

//------------------------------------------------------------------------------
// strided2
template<template<typename U> class Op, typename T, size_t Rank>
__global__ void strided2(
    const T *a, const Index<Rank> aIndex,
    const T *b, const Index<Rank> bIndex,
    T *out, const Index<Rank> oIndex,
    unsigned count
) {
    Op<T> op;
    // GRID_LOOP_STRIDED(ai, strideA, bi, strideB, oi, count) {
    //     out[oi] =  op(a[ai], b[bi]);
    // }
}

//------------------------------------------------------------------------------
// resolveType
// this function is for dense tensors that can be flattened where
// `isBufferIterable == true`, so strides must equal 0 or 1
template<template<typename U> class Op>
static cudaError_t resolveType(
    const void *a, const srtTensorDescriptor& aDesc,
    const void *b, const srtTensorDescriptor& bDesc,
    void *out, const srtTensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // make sure sizes fit within Cuda limitations
    size_t count = oDesc.count;
    assert(count <= UINT32_MAX);
    assert(aDesc.type == bDesc.type == oDesc.type);

    KernelPreCheck(stream);

    // unsigned blocks = BLOCK_COUNT(count);
    // unsigned threads = THREADS_PER_BLOCK;

    switch(aDesc.type) {
        // case CUDA_R_32F: strided2<<<blocks, threads, 0, stream>>>(a, ); break;
    //     // *** Figure out how to define operator+ for this type so it works in emulation mode
    //     // case CUDA_R_16BF: {
    //     //     unsigned n = shiftDownRoundingUp(countC, 1);
    //     //     addScalar_bfloat162<__nv_bfloat162><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, pScalar, c, n);
    //     //     break;
    //     // }
    //     case CUDA_R_16F: {
    //         unsigned n = shiftDownRoundingUp(count, 1);
    //         combine<Add, __half>(BLOCK_COUNT(n), threads, a, strideA, b, strideB, out, count, stream); break;
    //         break;
    //     }
    //     case CUDA_R_8I:  combine<Op, int8_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
    //     case CUDA_R_8U:  combine<Op, uint8_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
    //     case CUDA_R_16I: combine<Op, int16_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
    //     case CUDA_R_16U: combine<Op, uint16_t>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
    //     case CUDA_R_64F: combine<Op, double>(blocks, threads, a, strideA, b, strideB, out, count, stream); break;
        default: printf("cudaDataType_t not implemented"); exit(1);
    }

    return KernelPostCheck(stream);
}

//==============================================================================
// Swift importable C interface functions
// performs the operation with fully strided index calculations
//==============================================================================

//------------------------------------------------------------------------------
// strStridedAdd
cudaError_t strStridedAdd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return resolveType<Add>(a, *aDesc, b, *bDesc, out, *oDesc, stream);
}

