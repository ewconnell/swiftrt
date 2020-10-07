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
#include "srt_types.h"
#include "complex.h"
#include "index.h"

//==============================================================================
/// Op2
/// - Parameters:
///  - OpName: the operator instance name
///  - name: the name of the function this operator maps to,
///    for example: sin, cos, etc...
///  - conformance: a constant expression used to define which
///    type combinations are valid with the operator

// `packed` is a version of the operator where types smaller than 32 bit
// are retyped into packed versions to use with gpu SIMD instructions
#define Op2(OpName, name, conformance) \
template<typename _A, typename _B, typename _O> struct OpName { \
    typedef _A A; typedef _B B; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<B>() && \
                  isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __CUDA_DEVICE__ inline static void op(const A& a, const B& b, Out& out) { \
        if constexpr (conforms()) out = name(a, b); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,B>::type PB; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,PB,POut> packed; \
};

#define Op2SwapAB(OpName, name, conformance) \
template<typename _A, typename _B, typename _O> struct OpName { \
    typedef _A A; typedef _B B; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<B>() && \
                  isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __CUDA_DEVICE__ inline static void op(const A& a, const B& b, Out& out) { \
        if constexpr (conforms()) out = name(b, a); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,B>::type PB; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,PB,POut> packed; \
};

//==============================================================================
// kernels
//==============================================================================

template<typename Op, typename IndexA, typename IndexB, typename IndexO>
__global__ void mapAB(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int ib = indexB.linear(position);
        const int io = indexO.linear(position);
        Op::op(a[ia], b[ib], out[io]);
    }
}

//------------------------------------------------------------------------------
// tensorA Element
template<typename Op, typename IndexA, typename IndexO>
__global__ void mapAE(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::A element,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int io = indexO.linear(position);
        Op::op(a[ia], element, out[io]);
    }
}

//==============================================================================
// flattened

//--------------------------------------
// tensorA tensorB
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using B = const typename Op::B;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAB<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using E = const typename Op::B;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        E  e = *static_cast<E*>(pElement);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAE<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), e, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex

// tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using B = const typename Op::B;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    B* b = static_cast<B*>(pB);
    Out* out = static_cast<Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAB<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        b, IndexB(bDesc),
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// tensorA Element
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using E = const typename Op::A;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    E  e = *static_cast<E*>(pElement);
    Out* out = static_cast<Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAE<Op,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        e, 
        out, IndexO(oDesc));
    return cudaSuccess;
}

//==============================================================================
// selectRank

// tensorA tensorB
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexB,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexB<1>,IndexO<1>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexB<2>,IndexO<2>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexA<3>,IndexO<3>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// tensorA Element
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexO<1>>(a, aDesc, e, out, oDesc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexO<2>>(a, aDesc, e, out, oDesc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexO<3>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIndex

// tensorA tensorB
template<typename Op>
static inline cudaError_t selectIndex(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        if (aDesc.order == bDesc.order && aDesc.order == oDesc.order &&
            aDesc.isDense() && bDesc.isDense() && oDesc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>(a, aDesc, b, bDesc, out, oDesc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided>(a, aDesc, b, bDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element
template<typename Op>
static inline cudaError_t selectIndex(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
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
            return flattened<typename Op::packed>(a, aDesc, e, out, oDesc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided>(a, aDesc, e, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// selectOut

// tensorA tensorB
template<template<typename A, typename B, typename O> class Op, typename A, typename B>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIndex<Op<A,B,float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<A,B,float16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<A,B,bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<A,B,double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<A,B,int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<A,B,uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<A,B,int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<A,B,uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<A,B,int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<A,B,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<A,B,complexf>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// tensorA Element
template<template<typename A, typename B, typename O> class Op, typename A>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIndex<Op<A,A,float>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIndex<Op<A,A,float16>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIndex<Op<A,A,bfloat16>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIndex<Op<A,A,double>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIndex<Op<A,A,int32_t>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIndex<Op<A,A,uint8_t>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIndex<Op<A,A,int8_t>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIndex<Op<A,A,uint16_t>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIndex<Op<A,A,int16_t>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIndex<Op<A,A,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIndex<Op<A,A,complexf>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select
// converts from dynamic to static type and delegates for stride selection

// input and output are the same type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float,float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,float16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,complexf>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output are the same type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float,float>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,float16>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bfloat16>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,double>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,int32_t>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,uint8_t>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,int8_t>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,uint16_t>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,int16_t>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,complexf>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output are the different type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t selectTT_Bool(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type && oDesc.type == boolean);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output are the different type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t selectTT_Bool(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(oDesc.type == boolean);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bool>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,bool>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,bool>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
