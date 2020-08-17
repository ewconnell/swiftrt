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

template<typename F, typename E, int R,
         typename IndexA, typename IndexB, typename IndexO>
__global__ void mapAB(
    const E *a, IndexA indexA, 
    const E *b, IndexB indexB,
    E *out, IndexO indexO
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        out[io] = F::op(a[ia], b[ib]);
    }
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

template<
    typename F, typename E, int R,
    template<int U> class IndexA,
    template<int U> class IndexB,
    template<int U> class IndexO>
static cudaError_t mapIndex(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    auto tile = tileSize<R>(oDesc);
    auto grid = gridSize<R>(oDesc, tile);

    mapAB<F,E,R,IndexA<R>,IndexB<R>,IndexO<R>><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc), 
        b, IndexB<R>(bDesc),
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// selectIndex
/// invokes the correct kernel to selectIndex the elements of the two tensors
/// handling the cases of elements and single single sets.
///
template<template<typename U> class Op, typename E>
static cudaError_t selectIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef Op<E> F;
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);
    const E* b = static_cast<const E*>(pB);

    if (bDesc.isSingle()) {
        if (aDesc.isSingle()) {
            // single op single --> flat
            return mapIndex<F,E,1,Single,Single,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);

        } else if (aDesc.isDense()) {
            // flat op single --> flat
            return mapIndex<F,E,1,Flat,Single,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);

        } else {
            // strided op single --> flat
            switch (oDesc.rank) {
            case 1: return mapIndex<F,E,1,Strided,Single,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return mapIndex<F,E,2,Strided,Single,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return mapIndex<F,E,3,Strided,Single,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else if (bDesc.isDense()) {
        if (aDesc.isSingle()) {
            // single op flat --> flat
            return mapIndex<F,E,1,Single,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);

        } else if (aDesc.isDense()) {
            // dense op dense --> flat
            return mapIndex<F,E,1,Flat,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);

        } else {
            // strided op flat --> flat
            switch (oDesc.rank) {
            case 1: return mapIndex<F,E,1,Strided,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return mapIndex<F,E,2,Strided,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return mapIndex<F,E,3,Strided,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        if (aDesc.isSingle()) {
            // single op strided --> flat
            switch (oDesc.rank) {
            case 1: return mapIndex<F,E,1,Single,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return mapIndex<F,E,2,Single,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return mapIndex<F,E,3,Single,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        } else if (aDesc.isDense()) {
            // flat op strided --> flat
            switch (oDesc.rank) {
            case 1: return mapIndex<F,E,1,Flat,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return mapIndex<F,E,2,Flat,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return mapIndex<F,E,3,Flat,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        } else {
            // strided op strided --> flat
            switch (oDesc.rank) {
            case 1: return mapIndex<F,E,1,Strided,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return mapIndex<F,E,2,Strided,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return mapIndex<F,E,3,Strided,Strided,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
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
        case CUDA_R_32F:  return selectIndex<Op, float>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16BF: return selectIndex<Op, __nv_bfloat16>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16F:  return selectIndex<Op, __half>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_8I:   return selectIndex<Op, int8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_8U:   return selectIndex<Op, uint8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16I:  return selectIndex<Op, int16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16U:  return selectIndex<Op, uint16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_64F:  return selectIndex<Op, double>(a, aDesc, b, bDesc, out, oDesc, stream);
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
