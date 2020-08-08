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
/// selectRank
/// invokes the correct kernel to combine the elements of the two tensors
/// handling the cases of elements and single scalar sets.
///
template<template<typename U> class Op, typename T>
static void selectRank(
    const void* pA, const srtTensorDescriptor& aDesc,
    const void* pB, const srtTensorDescriptor& bDesc,
    void* pOut, const srtTensorDescriptor& oDesc,
    cudaStream_t stream,
    unsigned shiftCount = 0 
) {
    const T* a = static_cast<const T*>(pA);
    const T* b = static_cast<const T*>(pB);
    T* out = static_cast<T*>(pOut);

    // the count is divided in cases where values are handled as short vectors
    unsigned count = shiftDownRoundingUp(oDesc.count, shiftCount);

    // make sure total count fits within Cuda limitations
    assert(count <= UINT32_MAX);
    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;
    
    if (aDesc.spanCount == 1) {
        if (bDesc.spanCount == 1) {
            // scalar with scalar
            scalar2<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count); 

        } else if (isDense(bDesc)) {
            // scalar with flattened dense elements
            scalarTensor<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count); 
        } else {
            // fully indexed
        }
    } else {
        if (bDesc.spanCount == 1) {
            if (isDense(aDesc)) {
                // flattened dense elements with a scalar
                tensorScalar<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count); 
            } else {
                // A fully indexed
            }
        } else {
            // two sets of elements
            tensor2<Op, T><<<blocks, threads, 0, stream>>>(a, b, out, count);
        }
    }
}

//------------------------------------------------------------------------------
// selectType
// converts from dynamic to static type and delegates for stride selection
template<template<typename U> class Op>
static cudaError_t selectType(
    const void* a, const srtTensorDescriptor& aDesc,
    const void* b, const srtTensorDescriptor& bDesc,
    void* out, const srtTensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // must be same data type and rank, and output is dense
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type);
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    assert(oDesc.count == oDesc.spanCount);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == bDesc.order && aDesc.order == oDesc.order);

    KernelPreCheck(stream);
    switch(aDesc.type) {
        case CUDA_R_32F:  selectRank<Op, float>(a, aDesc, b, bDesc, out, oDesc, stream); break;
        // case CUDA_R_16BF: selectRank<Op, __nv_bfloat162>(a, aDesc, b, bDesc, out, oDesc, stream, 1); break;
        case CUDA_R_16F:  selectRank<Op, __half>(a, aDesc, b, bDesc, out, oDesc, stream, 1); break;
        case CUDA_R_8I:   selectRank<Op, int8_t>(a, aDesc, b, bDesc, out, oDesc, stream); break;
        case CUDA_R_8U:   selectRank<Op, uint8_t>(a, aDesc, b, bDesc, out, oDesc, stream); break;
        case CUDA_R_16I:  selectRank<Op, int16_t>(a, aDesc, b, bDesc, out, oDesc, stream); break;
        case CUDA_R_16U:  selectRank<Op, uint16_t>(a, aDesc, b, bDesc, out, oDesc, stream); break;
        case CUDA_R_64F:  selectRank<Op, double>(a, aDesc, b, bDesc, out, oDesc, stream); break;
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
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Add>(a, *aDesc, b, *bDesc, out, *oDesc, stream);
}

//------------------------------------------------------------------------------
// srtSub
cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Sub>(a, *aDesc, b, *bDesc, out, *oDesc, stream);
}

//------------------------------------------------------------------------------
// srtMul
cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Mul>(a, *aDesc, b, *bDesc, out, *oDesc, stream);
}

//------------------------------------------------------------------------------
// srtDiv
cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectType<Div>(a, *aDesc, b, *bDesc, out, *oDesc, stream);
}

