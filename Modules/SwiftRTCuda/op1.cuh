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
#pragma once
#include "float16.cuh"
#include "bfloat16.cuh"
#include "complex.h"
#include "index.cuh"

//==============================================================================
/// Op1
/// - Parameters:
///  - OpName: the operator instance name
///  - name: the name of the function this operator maps to,
///    for example: sin, cos, etc...
///  - swapAB: if `true` A and B will be swapped before passing them
///    to function `name`
///  - conformance: a constant expression used to define which
///    type combinations are valid with the operator

// `packed` is a version of the operator where types smaller than 32 bit
// are retyped into packed versions to use with gpu SIMD instructions
#define Op1(OpName, name, conformance) \
template<typename _A, typename _O> struct OpName { \
    typedef _A A; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __device__ static inline void op(const A& a, Out& out) { \
        if constexpr (conforms()) out = name(a); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,POut> packed; \
};

//==============================================================================
// kernels
//==============================================================================

template<typename Op, typename IndexA, typename IndexO>
__global__ void mapA(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int io = indexO.linear(position);
        Op::op(a[ia], out[io]);
    }
}

//==============================================================================
/// flattened
template<typename Op>
static inline cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    Out* out = static_cast<Out*>(pOut);

    // get tile and grid size for launch
    int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<Op,Flat,Flat><<<grid, tile, 0, stream>>>
        (a, Flat(aDesc), out, Flat(oDesc));
    return cudaSuccess;
}

//==============================================================================
// initIndex tensorA
template<typename Op, typename IndexA, typename IndexO>
static inline cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    Out* out = static_cast<Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapA<Op,IndexA,IndexO><<<grid, tile, 0, stream>>>
        (a, IndexA(aDesc), out, IndexO(oDesc));
    return cudaSuccess;
}

//==============================================================================
// selectRank
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexO<1>>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexO<2>>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexO<3>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIndex
template<typename Op>
static inline cudaError_t selectIndex(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        if (aDesc.order == oDesc.order && aDesc.isDense() && oDesc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>(a, aDesc, out, oDesc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided>(a, aDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// selectOut
template<template<typename T, typename O> class Op, typename A>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIndex<Op<A,float>>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<A,float16>>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<A,bfloat16>>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<A,double>>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<A,int32_t>>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<A,uint8_t>>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<A,int8_t>>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<A,uint16_t>>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<A,int16_t>>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<A,bool>>(a, aDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<A,complexf>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select
// converts from dynamic to static type and delegates for stride selection

// input and output are the same type
template<template<typename A, typename O> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float>>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16>>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16>>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double>>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t>>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t>>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t>>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t>>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t>>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool>>(a, aDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output can be different type
// like for casting or Complex Abs
template<template<typename A, typename O> class Op>
static inline cudaError_t selectT_O(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectOut<Op, float16>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectOut<Op, bfloat16>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectOut<Op, double>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uint8_t>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectOut<Op, int8_t>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectOut<Op, uint16_t>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectOut<Op, int16_t>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectOut<Op, bool>(a, aDesc, out, oDesc, stream);
    case complex32F: return selectOut<Op, complexf>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
