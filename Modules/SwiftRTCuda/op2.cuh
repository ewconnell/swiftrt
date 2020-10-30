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
#include "srt_traits.cuh"
#include "float16.cuh"
#include "bfloat16.cuh"
#include "complex.cuh"
#include "iterators.cuh"

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
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, Out& out) const { \
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
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, Out& out) const { \
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

template<typename Op, typename IterA, typename IterB, typename IterOut>
__global__ void map(const Op op, const IterA iterA, const IterB iterB, IterOut iterOut) {
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) op(iterA[p], iterB[p], iterOut[p]);
}

//==============================================================================
// flattened

//--------------------------------------
// tensorA tensorB
template<typename Op>
static cudaError_t flattened(
    const void* pA,
    const void* pB,
    void* pOut,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using B = const typename Op::B;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        Out* out = static_cast<Out*>(pOut);

        auto iterA = Flat(a, count);
        auto iterB = Flat(b, count);
        auto iterO = Flat(out, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterO.count);
        dim3 grid = gridSize(iterO.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterO);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element
template<typename Op>
static cudaError_t flattenedTE(
    const void* pA,
    const void* pElement,
    void* pOut,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using E = const typename Op::A;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        E  e = *static_cast<E*>(pElement);
        Out* out = static_cast<Out*>(pOut);

        auto iterA = Flat(a, count);
        auto iterE = Constant<E, 1>(e);
        auto iterO = Flat(out, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterO.count);
        dim3 grid = gridSize(iterO.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterO);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex

// tensorA tensorB
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterO>
static inline cudaError_t initIndex(
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

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterB = IterB<B*, Rank>(b, bDesc);
    auto iterO = IterO<Out*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterO);
    return cudaSuccess;
}

//--------------------------------------
// tensorA Element
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterO>
static inline cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using E = const typename Op::B;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    E  e = *static_cast<E*>(pElement);
    Out* out = static_cast<Out*>(pOut);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterE = Constant<E, Rank>(e);
    auto iterO = IterO<Out*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterO);
    return cudaSuccess;
}

//==============================================================================
// selectRank

// tensorA tensorB
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,1,IterA,IterB,IterO>(a, aDesc, b, bDesc, out, oDesc, stream);
    case 2: return initIndex<Op,2,IterA,IterB,IterO>(a, aDesc, b, bDesc, out, oDesc, stream);
    case 3: return initIndex<Op,3,IterA,IterB,IterO>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// tensorA Element
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,1,IterA,IterO>(a, aDesc, e, out, oDesc, stream);
    case 2: return initIndex<Op,2,IterA,IterO>(a, aDesc, e, out, oDesc, stream);
    case 3: return initIndex<Op,3,IterA,IterO>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIter

// tensorA tensorB
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided>(a, aDesc, b, bDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
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
    case real32F:  return selectIter<Op<A,B,float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<A,B,half>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<A,B,bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<A,B,double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<A,B,int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<A,B,uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<A,B,int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<A,B,uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<A,B,int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<A,B,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<A,B,Complex<float>>>(a, aDesc, b, bDesc, out, oDesc, stream);
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
    case real32F:  return selectIter<Op<A,A,float>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIter<Op<A,A,half>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIter<Op<A,A,bfloat16>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIter<Op<A,A,double>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIter<Op<A,A,int32_t>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIter<Op<A,A,uint8_t>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIter<Op<A,A,int8_t>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIter<Op<A,A,uint16_t>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIter<Op<A,A,int16_t>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIter<Op<A,A,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIter<Op<A,A,Complex<float>>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select flat
// converts from dynamic to static type and delegates for stride selection

// input and output are the same type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t select(
    srtDataType abtype,
    const void* a,
    const void* b,
    srtDataType otype,
    void* out,
    uint32_t count,
    cudaStream_t stream
) {
    if (abtype == otype) {
        switch(abtype) {
        case real32F:  return flattened<Op<float,float,float>>(a, b, out, count, stream);
        case real64F:  return flattened<Op<double,double,double>>(a, b, out, count, stream);
        case real32I:  return flattened<Op<int32_t,int32_t,int32_t>>(a, b, out, count, stream);
        
        case real16F:  return flattened<typename Op<half,half,half>::packed>(a, b, out, count, stream);
        case real16BF: return flattened<typename Op<bfloat16,bfloat16,bfloat16>::packed>(a, b, out, count, stream);
        case real8U:   return flattened<typename Op<uint8_t,uint8_t,uint8_t>::packed>(a, b, out, count, stream);
        case real8I:   return flattened<typename Op<int8_t,int8_t,int8_t>::packed>(a, b, out, count, stream);
        case real16U:  return flattened<typename Op<uint16_t,uint16_t,uint16_t>::packed>(a, b, out, count, stream);
        case real16I:  return flattened<typename Op<int16_t,int16_t,int16_t>::packed>(a, b, out, count, stream);
        case boolean:  return flattened<typename Op<bool,bool,bool>::packed>(a, b, out, count, stream);

        case complex32F: return flattened<Op<Complex<float>,Complex<float>,Complex<float>>>(a, b, out, count, stream);
        case complex16F: return flattened<Op<Complex<float16>,Complex<float16>,Complex<float16>>>(a, b, out, count, stream);
        case complex16BF: return flattened<Op<Complex<bfloat16>,Complex<bfloat16>,Complex<bfloat16>>>(a, b, out, count, stream);
        default: return cudaErrorNotSupported;
        }
    } else if (otype == boolean) {
        switch(abtype) {
        case real32F:  return flattened<Op<float,float,bool>>(a, b, out, count, stream);
        case real64F:  return flattened<Op<double,double,bool>>(a, b, out, count, stream);
        case real32I:  return flattened<Op<int32_t,int32_t,bool>>(a, b, out, count, stream);

        case real16F:  return flattened<typename Op<half,half,bool>::packed>(a, b, out, count, stream);
        case real16BF: return flattened<typename Op<bfloat16,bfloat16,bool>::packed>(a, b, out, count, stream);
        case real8U:   return flattened<typename Op<uint8_t,uint8_t,bool>::packed>(a, b, out, count, stream);
        case real8I:   return flattened<typename Op<int8_t,int8_t,bool>::packed>(a, b, out, count, stream);
        case real16U:  return flattened<typename Op<uint16_t,uint16_t,bool>::packed>(a, b, out, count, stream);
        case real16I:  return flattened<typename Op<int16_t,int16_t,bool>::packed>(a, b, out, count, stream);
        case boolean:  return flattened<typename Op<bool,bool,bool>::packed>(a, b, out, count, stream);

        case complex32F: return flattened<Op<Complex<float>,Complex<float>,bool>>(a, b, out, count, stream);
        case complex16F: return flattened<Op<Complex<float16>,Complex<float16>,bool>>(a, b, out, count, stream);
        case complex16BF: return flattened<Op<Complex<bfloat16>,Complex<bfloat16>,bool>>(a, b, out, count, stream);
        default: return cudaErrorNotSupported;
        }
    }
    return cudaErrorNotSupported;
}

// input and output are the same type
template<template<typename A, typename B, typename O> class Op>
static inline cudaError_t selectTE(
    srtDataType abtype,
    const void* a,
    const void* e,
    srtDataType otype,
    void* out,
    uint32_t count,
    cudaStream_t stream
) {
    if (abtype == otype) {
        switch(abtype) {
        case real32F:  return flattenedTE<Op<float,float,float>>(a, e, out, count, stream);
        case real64F:  return flattenedTE<Op<double,double,double>>(a, e, out, count, stream);
        case real32I:  return flattenedTE<Op<int32_t,int32_t,int32_t>>(a, e, out, count, stream);
        
        case real16F:  return flattenedTE<typename Op<half,half,half>::packed>(a, e, out, count, stream);
        case real16BF: return flattenedTE<typename Op<bfloat16,bfloat16,bfloat16>::packed>(a, e, out, count, stream);
        case real8U:   return flattenedTE<typename Op<uint8_t,uint8_t,uint8_t>::packed>(a, e, out, count, stream);
        case real8I:   return flattenedTE<typename Op<int8_t,int8_t,int8_t>::packed>(a, e, out, count, stream);
        case real16U:  return flattenedTE<typename Op<uint16_t,uint16_t,uint16_t>::packed>(a, e, out, count, stream);
        case real16I:  return flattenedTE<typename Op<int16_t,int16_t,int16_t>::packed>(a, e, out, count, stream);
        case boolean:  return flattenedTE<typename Op<bool,bool,bool>::packed>(a, e, out, count, stream);

        case complex32F: return flattenedTE<Op<Complex<float>,Complex<float>,Complex<float>>>(a, e, out, count, stream);
        case complex16F: return flattenedTE<Op<Complex<float16>,Complex<float16>,Complex<float16>>>(a, e, out, count, stream);
        case complex16BF: return flattenedTE<Op<Complex<bfloat16>,Complex<bfloat16>,Complex<bfloat16>>>(a, e, out, count, stream);
        default: return cudaErrorNotSupported;
        }
    } else if (otype == boolean) {
        switch(abtype) {
        case real32F:  return flattenedTE<Op<float,float,bool>>(a, e, out, count, stream);
        case real64F:  return flattenedTE<Op<double,double,bool>>(a, e, out, count, stream);
        case real32I:  return flattenedTE<Op<int32_t,int32_t,bool>>(a, e, out, count, stream);

        case real16F:  return flattenedTE<typename Op<half,half,bool>::packed>(a, e, out, count, stream);
        case real16BF: return flattenedTE<typename Op<bfloat16,bfloat16,bool>::packed>(a, e, out, count, stream);
        case real8U:   return flattenedTE<typename Op<uint8_t,uint8_t,bool>::packed>(a, e, out, count, stream);
        case real8I:   return flattenedTE<typename Op<int8_t,int8_t,bool>::packed>(a, e, out, count, stream);
        case real16U:  return flattenedTE<typename Op<uint16_t,uint16_t,bool>::packed>(a, e, out, count, stream);
        case real16I:  return flattenedTE<typename Op<int16_t,int16_t,bool>::packed>(a, e, out, count, stream);
        case boolean:  return flattenedTE<typename Op<bool,bool,bool>::packed>(a, e, out, count, stream);

        case complex32F: return flattenedTE<Op<Complex<float>,Complex<float>,bool>>(a, e, out, count, stream);
        case complex16F: return flattenedTE<Op<Complex<float16>,Complex<float16>,bool>>(a, e, out, count, stream);
        case complex16BF: return flattenedTE<Op<Complex<bfloat16>,Complex<bfloat16>,bool>>(a, e, out, count, stream);
        default: return cudaErrorNotSupported;
        }
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// select strided
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
    case real32F:  return selectIter<Op<float,float,float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,half>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,Complex<float>>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex16F: return selectIter<Op<Complex<float16>,Complex<float16>,Complex<float16>>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex16BF: return selectIter<Op<Complex<bfloat16>,Complex<bfloat16>,Complex<bfloat16>>>(a, aDesc, b, bDesc, out, oDesc, stream);
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
    case real32F:  return selectIter<Op<float,float,float>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,half>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bfloat16>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,double>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,int32_t>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,uint8_t>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,int8_t>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,uint16_t>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,int16_t>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,Complex<float>>>(a, aDesc, e, out, oDesc, stream);
    case complex16F: return selectIter<Op<Complex<float16>,Complex<float16>,Complex<float16>>>(a, aDesc, e, out, oDesc, stream);
    case complex16BF: return selectIter<Op<Complex<bfloat16>,Complex<bfloat16>,Complex<bfloat16>>>(a, aDesc, e, out, oDesc, stream);
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
    case real32F:  return selectIter<Op<float,float,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex16F: return selectIter<Op<Complex<float16>,Complex<float16>,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex16BF: return selectIter<Op<Complex<bfloat16>,Complex<bfloat16>,bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
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
    case real32F:  return selectIter<Op<float,float,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bool>>(a, aDesc, e, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,bool>>(a, aDesc, e, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,bool>>(a, aDesc, e, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex16F: return selectIter<Op<Complex<float16>,Complex<float16>,bool>>(a, aDesc, e, out, oDesc, stream);
    case complex16BF: return selectIter<Op<Complex<bfloat16>,Complex<bfloat16>,bool>>(a, aDesc, e, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
