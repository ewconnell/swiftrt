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
// kernels
//==============================================================================

//------------------------------------------------------------------------------
// scalar2
template<template<typename U> class Op, typename T>
__global__ void scalar2(const T *a, const T *b, T *out, unsigned count) {
    Op<T> op;
    T result = op(a[0], b[0]);
    GRID_LOOP(i, count) {
        out[i] = result;
    }
}

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
template<template<typename U> class Op, int Rank, typename T>
static void combine(
    const void* pA, const srtTensorDescriptor& aDesc,
    const void* pB, const srtTensorDescriptor& bDesc,
    void* pOut, const srtTensorDescriptor& oDesc,
    cudaStream_t stream,
    unsigned shiftCount = 0 
) {
    const T* a = static_cast<const T*>(pA);
    const T* b = static_cast<const T*>(pB);
    T* out = static_cast<T*>(pOut);

}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
void srtAddR1Float(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    // combine<Add, 1, float>(a, )
}

