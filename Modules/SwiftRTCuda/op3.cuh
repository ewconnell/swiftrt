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

#include <stdio.h>

//==============================================================================
/// Op3
/// - Parameters:
///  - OpName: the operator instance name
///  - name: the name of the function this operator maps to,
///    for example: sin, cos, etc...
///  - swapAB: if `true` A and B will be swapped before passing them
///    to function `name`
///  - conformance: a constant expression used to define which
///    type combinations are valid with the operator

// Note: `packed` is a version of the operator where types smaller than 32 bit
// are retyped into packed versions to use with gpu SIMD instructions

//--------------------------------------
#define Op3Same(OpName, name, conformance) \
template<typename T> struct OpName { \
    typedef T A; typedef T B; typedef T C; typedef T Out; \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, const C& c, Out& out) const { \
        if constexpr (conforms()) out = name(a, b, c); \
    } \
    typedef typename packed<T>::type PT; \
    typedef OpName<PT> packed; \
};

//--------------------------------------
#define Op3(OpName, name, conformance) \
template<typename _A, typename _B, typename _C, typename _O> struct OpName { \
    typedef _A A; typedef _B B; typedef _C C; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<B>() && \
                  isPacked<A>() == isPacked<C>() && \
                  isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, const C& c, Out& out) const { \
        if constexpr (conforms()) out = name(a, b, c); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,B>::type PB; \
    typedef typename matching_packed<PA,C>::type PC; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,PB,PC,POut> packed; \
};

//--------------------------------------
#define Op3SwapBC(OpName, name, conformance) \
template<typename _A, typename _B, typename _C, typename _O> struct OpName { \
    typedef _A A; typedef _B B; typedef _C C; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<B>() && \
                  isPacked<A>() == isPacked<C>() && \
                  isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, const C& c, Out& out) const { \
        if constexpr (conforms()) out = name(a, c, b); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,B>::type PB; \
    typedef typename matching_packed<PA,C>::type PC; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,PB,PC,POut> packed; \
};

//--------------------------------------
#define Op3SwapBCSame(OpName, name, conformance) \
template<typename T> struct OpName { \
    typedef T A; typedef T B; typedef T C; typedef T Out; \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, const C& c, Out& out) const { \
        if constexpr (conforms()) out = name(a, c, b); \
    } \
    typedef typename packed<T>::type PT; \
    typedef OpName<PT> packed; \
};

//--------------------------------------
// this is for vjp min/max
#define Op32(OpName, name, conformance) \
template<typename T> struct OpName { \
    typedef T A; typedef T B; typedef T C; typedef T Out; \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ void operator()(const A& a, const B& b, const C& c, Out& out0, Out& out1) const { \
        if constexpr (conforms()) name(a, b, c, out0, out1); \
    } \
    typedef typename packed<T>::type PT; \
    typedef OpName<PT> packed; \
};

//==============================================================================
// kernels
//==============================================================================

//--------------------------------------
// tensorA tensorB tensorC Out
template<typename Op, typename IterA, typename IterB, typename IterC, typename IterOut>
__global__ void map(
    const Op op,
    const IterA iterA,
    const IterB iterB,
    const IterC iterC,
    IterOut iterOut
) {
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) op(iterA[p], iterB[p], iterC[p], iterOut[p]);

    // if (iterOut.isInBounds(p)) {
    //     auto ia = iterA.linear(p);
    //     auto ib = iterB.linear(p);
    //     auto ic = iterC.linear(p);
    //     auto io = iterOut.linear(p);
    //     op(iterA[p], iterB[p], iterC[p], iterOut[p]);
    //     // op(iterA[ia], iterB[ib], iterC[ic], iterOut[io]);
    // }
}

//--------------------------------------
// tensorA tensorB tensorC Out0 Out1
template<typename Op, typename IterA, typename IterB, typename IterC, typename IterOut>
__global__ void map(
    const Op op,
    const IterA iterA,
    const IterB iterB,
    const IterC iterC,
    IterOut iterOut0,
    IterOut iterOut1
) {
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut0.isInBounds(p)) op(iterA[p], iterB[p], iterC[p], iterOut0[p], iterOut0[p]);
}

//==============================================================================
// flattened

//--------------------------------------
// tensorA tensorB tensorC Out
template<typename Op>
static cudaError_t flattened(
    const void* pA,
    const void* pB,
    const void* pC,
    void* pOut,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using B = const typename Op::B;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        C* c = static_cast<C*>(pC);
        Out* out = static_cast<Out*>(pOut);

        auto iterA = Flat(a, count);
        auto iterB = Flat(b, count);
        auto iterC = Flat(c, count);
        auto iterO = Flat(out, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterO.count);
        dim3 grid = gridSize(iterO.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterC, iterO);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element tensorC Out
template<typename Op>
static cudaError_t flattenedTET(
    const void* pA,
    const void* pElement,
    const void* pC,
    void* pOut,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using E = const typename Op::A;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        E  e = *static_cast<E*>(pElement);
        C* c = static_cast<C*>(pC);
        Out* out = static_cast<Out*>(pOut);

        auto iterA = Flat(a, count);
        auto iterE = Constant<E, 1>(e);
        auto iterC = Flat(c, count);
        auto iterO = Flat(out, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterO.count);
        dim3 grid = gridSize(iterO.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterC, iterO);
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}


//--------------------------------------
// tensorA tensorB tensorC Out Out
template<typename Op>
static cudaError_t flattened(
    const void* pA,
    const void* pB,
    const void* pC,
    void* pOut0,
    void* pOut1,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using B = const typename Op::B;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        C* c = static_cast<C*>(pC);
        Out* out0 = static_cast<Out*>(pOut0);
        Out* out1 = static_cast<Out*>(pOut1);

        auto iterA = Flat(a, count);
        auto iterB = Flat(b, count);
        auto iterC = Flat(c, count);
        auto iterOut0 = Flat(out0, count);
        auto iterOut1 = Flat(out1, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterOut0.count);
        dim3 grid = gridSize(iterOut0.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterC, iterOut0, iterOut1);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element tensorC Out Out
template<typename Op>
static cudaError_t flattenedTET(
    const void* pA,
    const void* pElement,
    const void* pC,
    void* pOut0,
    void* pOut1,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using E = const typename Op::A;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        E  e = *static_cast<E*>(pElement);
        C* c = static_cast<C*>(pC);
        Out* out0 = static_cast<Out*>(pOut0);
        Out* out1 = static_cast<Out*>(pOut1);

        auto iterA = Flat(a, count);
        auto iterE = Constant<E, 1>(e);
        auto iterC = Flat(c, count);
        auto iterOut0 = Flat(out0, count);
        auto iterOut1 = Flat(out1, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterOut0.count);
        dim3 grid = gridSize(iterOut0.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterC, iterOut0, iterOut1);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex

//--------------------------------------
// tensorA tensorB tensorC Out
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using B = const typename Op::B;
    using C = const typename Op::C;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    B* b = static_cast<B*>(pB);
    C* c = static_cast<C*>(pC);
    Out* out = static_cast<Out*>(pOut);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterB = IterB<B*, Rank>(b, bDesc);
    auto iterC = IterC<C*, Rank>(c, cDesc);
    auto iterO = IterO<Out*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterC, iterO);
    return cudaSuccess;
}

//--------------------------------------
// tensorA Element tensorC Out
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    const void* pElement,
    const void* pC, const TensorDescriptor& cDesc, 
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using E = const typename Op::A;
    using C = const typename Op::C;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    E  e = *static_cast<E*>(pElement);
    C* c = static_cast<C*>(pC);
    Out* out = static_cast<Out*>(pOut);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterE = Constant<E, Rank>(e);
    auto iterC = IterC<C*, Rank>(c, cDesc);
    auto iterO = IterO<Out*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterC, iterO);
    return cudaSuccess;
}

//--------------------------------------
// tensorA tensorB tensorC Out0 Out1
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut0, const TensorDescriptor& oDesc0,
    void* pOut1, const TensorDescriptor& oDesc1,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using B = const typename Op::B;
    using C = const typename Op::C;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    B* b = static_cast<B*>(pB);
    C* c = static_cast<C*>(pC);
    Out* out0 = static_cast<Out*>(pOut0);
    Out* out1 = static_cast<Out*>(pOut1);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterB = IterB<B*, Rank>(b, bDesc);
    auto iterC = IterC<C*, Rank>(c, cDesc);
    auto iterOut0 = IterO<Out*, Rank>(out0, oDesc0);
    auto iterOut1 = IterO<Out*, Rank>(out1, oDesc1);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterOut0.shape);
    dim3 grid = gridSize<Rank>(iterOut1.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterB, iterC, iterOut0, iterOut1);
    return cudaSuccess;
}

//--------------------------------------
// tensorA Element tensorC Out Out
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    const void* pElement,
    const void* pC, const TensorDescriptor& cDesc, 
    void* pOut0, const TensorDescriptor& oDesc0,
    void* pOut1, const TensorDescriptor& oDesc1,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using E = const typename Op::A;
    using C = const typename Op::C;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    E  e = *static_cast<E*>(pElement);
    C* c = static_cast<C*>(pC);
    Out* out0 = static_cast<Out*>(pOut0);
    Out* out1 = static_cast<Out*>(pOut1);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterE = Constant<E, Rank>(e);
    auto iterC = IterC<C*, Rank>(c, cDesc);
    auto iterOut0 = IterO<Out*, Rank>(out0, oDesc0);
    auto iterOut1 = IterO<Out*, Rank>(out1, oDesc1);


    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterOut0.shape);
    dim3 grid = gridSize<Rank>(iterOut1.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterE, iterC, iterOut0, iterOut1);
    return cudaSuccess;
}

//==============================================================================
// selectRank

//--------------------------------------
// tensorA tensorB tensorC Out
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank &&
        aDesc.rank == cDesc.rank && aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,1,IterA,IterB,IterC,IterO>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case 2: return initIndex<Op,2,IterA,IterB,IterC,IterO>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case 3: return initIndex<Op,3,IterA,IterB,IterC,IterO>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element tensorC Out
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == cDesc.rank && aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,1,IterA,IterC,IterO>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case 2: return initIndex<Op,2,IterA,IterC,IterO>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case 3: return initIndex<Op,3,IterA,IterC,IterO>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA tensorB tensorC Out Out
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterB,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& oDesc0,
    void* out1, const TensorDescriptor& oDesc1,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == cDesc.rank &&
        aDesc.rank == oDesc0.rank && aDesc.rank == oDesc1.rank);
    switch(oDesc0.rank) {
    case 1: return initIndex<Op,1,IterA,IterB,IterC,IterO>
        (a, aDesc, b, bDesc, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    case 2: return initIndex<Op,2,IterA,IterB,IterC,IterO>
        (a, aDesc, b, bDesc, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    case 3: return initIndex<Op,3,IterA,IterB,IterC,IterO>
        (a, aDesc, b, bDesc, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element tensorC Out Out
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterC,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& oDesc0,
    void* out1, const TensorDescriptor& oDesc1,
    cudaStream_t stream
) {
    assert(aDesc.rank == cDesc.rank && aDesc.rank == oDesc0.rank);
    switch(oDesc0.rank) {
    case 1: return initIndex<Op,1,IterA,IterC,IterO>
        (a, aDesc, e, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    case 2: return initIndex<Op,2,IterA,IterC,IterO>
        (a, aDesc, e, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    case 3: return initIndex<Op,3,IterA,IterC,IterO>
        (a, aDesc, e, c, cDesc, out0, oDesc0, out1, oDesc1, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIter

// tensorA tensorB tensorC
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == cDesc.rank &&
           aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided,Strided>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element tensorC
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == cDesc.rank && aDesc.rank == oDesc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA tensorB tensorC Out Out
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == cDesc.rank &&
           aDesc.rank == o0Desc.rank && aDesc.rank == o1Desc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided,Strided>
            (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element tensorC Out Out
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    const void* b,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    assert(aDesc.rank == cDesc.rank && aDesc.rank == o0Desc.rank &&
           aDesc.rank == o1Desc.rank);
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided>
            (a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// selectOut

// tensorA tensorB tensorC
template<template<typename A, typename B, typename C, typename O> class Op, 
         typename A, typename B, typename C>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIter<Op<A,B,C,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<A,B,C,half>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<A,B,C,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<A,B,C,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<A,B,C,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<A,B,C,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<A,B,C,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<A,B,C,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<A,B,C,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<A,B,C,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<A,B,C,Complex<float>>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// tensorA Element tensorC
template<template<typename A, typename B, typename C, typename O> class Op,
         typename A, typename C>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIter<Op<A,A,C,float>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<A,A,C,half>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<A,A,C,bfloat16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<A,A,C,double>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<A,A,C,int32_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<A,A,C,uint8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<A,A,C,int8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<A,A,C,uint16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<A,A,C,int16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<A,A,C,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<A,A,C,Complex<float>>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select flat
// converts from dynamic to static type and delegates for stride selection

//--------------------------------------
// tensorA tensorB tensorC Out
// inputs and output are the same type
template<template<typename T> class Op>
static inline cudaError_t select(
    srtDataType type,
    const void* a,
    const void* b,
    const void* c,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    switch(type) {
    case real32F:  return flattened<Op<float>>(a, b, c, out, count, stream);
    case real64F:  return flattened<Op<double>>(a, b, c, out, count, stream);
    case real32I:  return flattened<Op<int32_t>>(a, b, c, out, count, stream);

    // recast types that are smaller than 32 bit to their packed simd form
    case real16F:  return flattened<typename Op<float16>::packed>(a, b, c, out, count, stream);
    case real16BF: return flattened<typename Op<bfloat16>::packed>(a, b, c, out, count, stream);
    case real8I:   return flattened<typename Op<int8_t>::packed>(a, b, c, out, count, stream);
    case real8U:   return flattened<typename Op<uint8_t>::packed>(a, b, c, out, count, stream);
    case real16I:  return flattened<typename Op<int16_t>::packed>(a, b, c, out, count, stream);
    case real16U:  return flattened<typename Op<uint16_t>::packed>(a, b, c, out, count, stream);
    case boolean:  return flattened<typename Op<bool>::packed>(a, b, c, out, count, stream);

    case complex32F:  return flattened<Op<Complex<float>>>(a, b, c, out, count, stream);
    case complex16F:  return flattened<Op<Complex<float16>>>(a, b, c, out, count, stream);
    case complex16BF: return flattened<Op<Complex<bfloat16>>>(a, b, c, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element tensorC Out
// inputs and output are the same type
template<template<typename T> class Op>
static inline cudaError_t selectTET(
    srtDataType type,
    const void* a,
    const void* element,
    const void* c,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    switch(type) {
    case real32F:  return flattenedTET<Op<float>>(a, element, c, out, count, stream);
    case real64F:  return flattenedTET<Op<double>>(a, element, c, out, count, stream);
    case real32I:  return flattenedTET<Op<int32_t>>(a, element, c, out, count, stream);

    // recast types that are smaller than 32 bit to their packed simd form
    case real16F:  return flattenedTET<typename Op<float16>::packed>(a, element, c, out, count, stream);
    case real16BF: return flattenedTET<typename Op<bfloat16>::packed>(a, element, c, out, count, stream);
    case real8I:   return flattenedTET<typename Op<int8_t>::packed>(a, element, c, out, count, stream);
    case real8U:   return flattenedTET<typename Op<uint8_t>::packed>(a, element, c, out, count, stream);
    case real16I:  return flattenedTET<typename Op<int16_t>::packed>(a, element, c, out, count, stream);
    case real16U:  return flattenedTET<typename Op<uint16_t>::packed>(a, element, c, out, count, stream);
    case boolean:  return flattenedTET<typename Op<bool>::packed>(a, element, c, out, count, stream);

    case complex32F:  return flattenedTET<Op<Complex<float>>>(a, element, c, out, count, stream);
    case complex16F:  return flattenedTET<Op<Complex<float16>>>(a, element, c, out, count, stream);
    case complex16BF: return flattenedTET<Op<Complex<bfloat16>>>(a, element, c, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}


//==============================================================================
// select
// converts from dynamic to static type and delegates for stride selection

//--------------------------------------
// tensorA tensorB tensorC
// input and output are the same type
template<template<typename A, typename B, typename C, typename O> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIter<Op<float,float,float,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,half,half>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bfloat16,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,double,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,int32_t,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,uint8_t,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,int8_t,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,uint16_t,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,int16_t,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,Complex<float>,Complex<float>>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element tensorC
// input and output are the same type
template<template<typename A, typename B, typename C, typename O> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        assert(aDesc.type == cDesc.type);
        switch(aDesc.type) {
        case real32F:  return selectIter<Op<float,float,float,float>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16F:  return selectIter<Op<half,half,half,half>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16BF: return selectIter<Op<bfloat16,bfloat16,bfloat16,bfloat16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real64F:  return selectIter<Op<double,double,double,double>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real32I:  return selectIter<Op<int32_t,int32_t,int32_t,int32_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real8U:   return selectIter<Op<uint8_t,uint8_t,uint8_t,uint8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real8I:   return selectIter<Op<int8_t,int8_t,int8_t,int8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16U:  return selectIter<Op<uint16_t,uint16_t,uint16_t,uint16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16I:  return selectIter<Op<int16_t,int16_t,int16_t,int16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case boolean:  return selectIter<Op<bool,bool,bool,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case complex32F: return selectIter<Op<Complex<float>,Complex<float>,Complex<float>,Complex<float>>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else if(oDesc.type == boolean) {
        assert(aDesc.type == cDesc.type);
        switch(aDesc.type) {
        case real32F:  return selectIter<Op<float,float,float,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16F:  return selectIter<Op<half,half,half,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16BF: return selectIter<Op<bfloat16,bfloat16,bfloat16,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real64F:  return selectIter<Op<double,double,double,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real32I:  return selectIter<Op<int32_t,int32_t,int32_t,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real8U:   return selectIter<Op<uint8_t,uint8_t,uint8_t,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real8I:   return selectIter<Op<int8_t,int8_t,int8_t,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16U:  return selectIter<Op<uint16_t,uint16_t,uint16_t,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case real16I:  return selectIter<Op<int16_t,int16_t,int16_t,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case boolean:  return selectIter<Op<bool,bool,bool,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        case complex32F: return selectIter<Op<Complex<float>,Complex<float>,Complex<float>,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// input and output are the same type
template<template<typename T> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectIter<Op<float>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16F:  return selectIter<Op<half>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16BF: return selectIter<Op<bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real64F:  return selectIter<Op<double>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real32I:  return selectIter<Op<int32_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8U:   return selectIter<Op<uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8I:   return selectIter<Op<int8_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16U:  return selectIter<Op<uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16I:  return selectIter<Op<int16_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case boolean:  return selectIter<Op<bool>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case complex32F: return selectIter<Op<Complex<float>>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// input and output are the same type
template<template<typename T> class Op>
static inline cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectIter<Op<float>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16F:  return selectIter<Op<half>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16BF: return selectIter<Op<bfloat16>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real64F:  return selectIter<Op<double>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real32I:  return selectIter<Op<int32_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8U:   return selectIter<Op<uint8_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8I:   return selectIter<Op<int8_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16U:  return selectIter<Op<uint16_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16I:  return selectIter<Op<int16_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case boolean:  return selectIter<Op<bool>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case complex32F: return selectIter<Op<Complex<float>>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA tensorB tensorC
// input and output are the different type
template<template<typename A, typename B, typename C, typename O> class Op>
static inline cudaError_t selectTTT_O(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // only call this function when the output doesn't match
    assert(aDesc.type == bDesc.type && aDesc.type != oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float,float,float>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectOut<Op, half,half,half>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectOut<Op, bfloat16,bfloat16,bfloat16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectOut<Op, double,double,double>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t,int32_t,int32_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uint8_t,uint8_t,uint8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectOut<Op, int8_t,int8_t,int8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectOut<Op, uint16_t,uint16_t,uint16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectOut<Op, int16_t,int16_t,int16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectOut<Op, bool,bool,bool>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectOut<Op, Complex<float>,Complex<float>,Complex<float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// tensorA tensorB tensorC

// input and output are same type, but C is Bool
template<template<typename A, typename B, typename C, typename O> class Op>
static inline cudaError_t selectTTBool_T(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // only call this function when the output doesn't match
    assert(aDesc.type == bDesc.type && cDesc.type == boolean &&
           aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIter<Op<float,float,bool,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half,bool,half>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16,bool,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double,bool,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t,bool,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t,bool,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t,bool,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t,bool,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t,bool,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool,bool,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIter<Op<Complex<float>,Complex<float>,bool,Complex<float>>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output are same type, but C is Bool
template<template<typename A, typename B, typename C, typename O> class Op>
static inline cudaError_t select(
    srtDataType type,
    const void* a,
    const void* b,
    srtDataType ctype,
    const void* c,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    assert(ctype == boolean);
    switch(type) {
    case real32F:  return flattened<Op<float,float,bool,float>>(a, b, c, out, count, stream);
    case real64F:  return flattened<Op<double,double,bool,double>>(a, b, c, out, count, stream);
    case real32I:  return flattened<Op<int32_t,int32_t,bool,int32_t>>(a, b, c, out, count, stream);

    case real16F:  return flattened<typename Op<half,half,bool,half>::packed>(a, b, c, out, count, stream);
    case real16BF: return flattened<typename Op<bfloat16,bfloat16,bool,bfloat16>::packed>(a, b, c, out, count, stream);
    case real8U:   return flattened<typename Op<uint8_t,uint8_t,bool,uint8_t>::packed>(a, b, c, out, count, stream);
    case real8I:   return flattened<typename Op<int8_t,int8_t,bool,int8_t>::packed>(a, b, c, out, count, stream);
    case real16U:  return flattened<typename Op<uint16_t,uint16_t,bool,uint16_t>::packed>(a, b, c, out, count, stream);
    case real16I:  return flattened<typename Op<int16_t,int16_t,bool,int16_t>::packed>(a, b, c, out, count, stream);
    case boolean:  return flattened<typename Op<bool,bool,bool,bool>::packed>(a, b, c, out, count, stream);

    case complex16F:  return flattened<Op<Complex<float16>,Complex<float16>,bool,Complex<float16>>>(a, b, c, out, count, stream);
    case complex16BF: return flattened<Op<Complex<bfloat16>,Complex<bfloat16>,bool,Complex<bfloat16>>>(a, b, c, out, count, stream);
    case complex32F:  return flattened<Op<Complex<float>,Complex<float>,bool,Complex<float>>>(a, b, c, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}
