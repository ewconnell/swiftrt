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
#include "index.cuh"

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
#define Op3(OpName, name, conformance) \
template<typename _A, typename _B, typename _C, typename _O> struct OpName { \
    typedef _A A; typedef _B B; typedef _C C; typedef _O Out; \
    static_assert(isPacked<A>() == isPacked<B>() && \
                  isPacked<A>() == isPacked<C>() && \
                  isPacked<A>() == isPacked<Out>(), "packed type mismatch"); \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ static void op(const A& a, const B& b, const C& c, Out& out) { \
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
    __DEVICE_INLINE__ static void op(const A& a, const B& b, const C& c, Out& out) { \
        if constexpr (conforms()) out = name(a, c, b); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,B>::type PB; \
    typedef typename matching_packed<PA,C>::type PC; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,PB,PC,POut> packed; \
};

//--------------------------------------
// this is for vjp min/max
#define Op32(OpName, name, conformance) \
template<typename T> struct OpName { \
    typedef T A; typedef T B; typedef T C; typedef T Out; \
    constexpr static bool conforms() { return (conformance); } \
    __DEVICE_INLINE__ static void op(const A& a, const B& b, const C& c, Out& out0, Out& out1) { \
        if constexpr (conforms()) name(a, b, c, out0, out1); \
    } \
    typedef typename packed<T>::type PT; \
    typedef OpName<PT> packed; \
};

//==============================================================================
// kernels
//==============================================================================

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op, typename IndexA, typename IndexB, typename IndexC, typename IndexO>
__global__ void mapABC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int ib = indexB.linear(position);
        const int ic = indexC.linear(position);
        const int io = indexO.linear(position);
        Op::op(a[ia], b[ib], c[ic], out[io]);
    }
}

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op, typename IndexA, typename IndexB, typename IndexC, typename IndexO>
__global__ void mapABC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out0, const IndexO indexO0,
    typename Op::Out* __restrict__ out1, const IndexO indexO1
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO0.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int ib = indexB.linear(position);
        const int ic = indexC.linear(position);
        const int io0 = indexO0.linear(position);
        const int io1 = indexO1.linear(position);
        Op::op(a[ia], b[ib], c[ic], out0[io0], out1[io1]);
    }
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op, typename IndexA, typename IndexC, typename IndexO>
__global__ void mapAEC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::A element,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int ic = indexC.linear(position);
        const int io = indexO.linear(position);
        Op::op(a[ia], element, c[ic], out[io]);
    }
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op, typename IndexA, typename IndexC, typename IndexO>
__global__ void mapAEC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::A element,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out0, const IndexO indexO0,
    typename Op::Out* __restrict__ out1, const IndexO indexO1
) {
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO0.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int ic = indexC.linear(position);
        const int io0 = indexO0.linear(position);
        const int io1 = indexO1.linear(position);
        Op::op(a[ia], element, c[ic], out0[io0], out1[io1]);
    }
}

//==============================================================================
// flattened

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using B = const typename Op::B;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        C* c = static_cast<C*>(pC);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapABC<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>(
            a, Flat(aDesc), 
            b, Flat(bDesc), 
            c, Flat(cDesc), 
            out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using E = const typename Op::B;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        E  e = *static_cast<E*>(pElement);
        C* c = static_cast<C*>(pC);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAEC<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), e, c, Flat(cDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut0, const TensorDescriptor& o0Desc,
    void* pOut1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using B = const typename Op::B;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        C* c = static_cast<C*>(pC);
        Out* out0 = static_cast<Out*>(pOut0);
        Out* out1 = static_cast<Out*>(pOut1);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(o0Desc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(o0Desc, tile);

        mapABC<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>(
            a, Flat(aDesc), 
            b, Flat(bDesc), 
            c, Flat(cDesc), 
            out0, Flat(o0Desc),
            out1, Flat(o1Desc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut0, const TensorDescriptor& o0Desc,
    void* pOut1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using C = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        A  e = *static_cast<A*>(pElement);
        C* c = static_cast<C*>(pC);
        Out* out0 = static_cast<Out*>(pOut0);
        Out* out1 = static_cast<Out*>(pOut1);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(o0Desc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(o0Desc, tile);

        mapAEC<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>(
            a, Flat(aDesc), 
            e,
            c, Flat(cDesc), 
            out0, Flat(o0Desc),
            out1, Flat(o1Desc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op, typename IndexA, typename IndexB, typename IndexC, typename IndexO>
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

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapABC<Op,IndexA,IndexB,IndexC,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        b, IndexB(bDesc),
        c, IndexC(cDesc),
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op, typename IndexA, typename IndexC, typename IndexO>
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

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAEC<Op,IndexA,IndexC,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        e,
        c, IndexC(cDesc),
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op, typename IndexA, typename IndexB, typename IndexC, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut0, const TensorDescriptor& o0Desc,
    void* pOut1, const TensorDescriptor& o1Desc,
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

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(o0Desc);
    dim3 grid = gridSize<IndexO::Rank>(o0Desc, tile);

    mapABC<Op,IndexA,IndexB,IndexC,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        b, IndexB(bDesc),
        c, IndexC(cDesc),
        out0, IndexO(o0Desc),
        out1, IndexO(o1Desc));
    return cudaSuccess;
}


//--------------------------------------
// tensorA Element tensorC
template<typename Op, typename IndexA, typename IndexC, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    const void* pC, const TensorDescriptor& cDesc,
    void* pOut0, const TensorDescriptor& o0Desc,
    void* pOut1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using C = const typename Op::C;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    A  e = *static_cast<A*>(pElement);
    C* c = static_cast<C*>(pC);
    Out* out0 = static_cast<Out*>(pOut0);
    Out* out1 = static_cast<Out*>(pOut1);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(o0Desc);
    dim3 grid = gridSize<IndexO::Rank>(o0Desc, tile);

    mapAEC<Op,IndexA,IndexC,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        e,
        c, IndexC(cDesc),
        out0, IndexO(o0Desc),
        out1, IndexO(o1Desc));
    return cudaSuccess;
}

//==============================================================================
// selectRank

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexB,
    template<int R> class IndexC,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexB<1>,IndexC<1>,IndexO<1>>
        (a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexB<2>,IndexC<2>,IndexO<2>>
        (a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexA<3>,IndexC<3>,IndexO<3>>
        (a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element tensorC
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexC,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* e,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexC<1>,IndexO<1>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexC<2>,IndexO<2>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexC<3>,IndexO<3>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexB,
    template<int R> class IndexC,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    assert(aDesc.rank == o0Desc.rank);
    switch(o0Desc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexB<1>,IndexC<1>,IndexO<1>>
        (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexB<2>,IndexC<2>,IndexO<2>>
        (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexA<3>,IndexC<3>,IndexO<3>>
        (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA tensorB tensorC
template<typename Op,
    template<int R> class IndexA,
    template<int R> class IndexC,
    template<int R> class IndexO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b,
    const void* c, const TensorDescriptor& cDesc,
    void* out0, const TensorDescriptor& o0Desc,
    void* out1, const TensorDescriptor& o1Desc,
    cudaStream_t stream
) {
    assert(aDesc.rank == o0Desc.rank);
    switch(o0Desc.rank) {
    case 1: return initIndex<Op,IndexA<1>,IndexC<1>,IndexO<1>>
        (a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case 2: return initIndex<Op,IndexA<2>,IndexC<2>,IndexO<2>>
        (a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case 3: return initIndex<Op,IndexA<3>,IndexC<3>,IndexO<3>>
        (a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIndex

// tensorA tensorB tensorC
template<typename Op>
static inline cudaError_t selectIndex(
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
        if (aDesc.order == bDesc.order && aDesc.order == cDesc.order && 
            aDesc.order == oDesc.order &&
            aDesc.isDense() && bDesc.isDense() &&
            cDesc.isDense() && oDesc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided,Strided>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element tensorC
template<typename Op>
static inline cudaError_t selectIndex(
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
        if (aDesc.order == oDesc.order && aDesc.order == cDesc.order &&
            aDesc.isDense() && cDesc.isDense() && oDesc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>(a, aDesc, e, c, cDesc, out, oDesc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA tensorB tensorC
template<typename Op>
static inline cudaError_t selectIndex(
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
        if (aDesc.order == bDesc.order && aDesc.order == cDesc.order && 
            aDesc.order == o0Desc.order && aDesc.order == o1Desc.order &&
            aDesc.isDense() && bDesc.isDense() && cDesc.isDense() &&
            o0Desc.isDense() && o1Desc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>
                (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
        }
        // TODO add support for tile based indexes
        return selectRank<Op,Strided,Strided,Strided,Strided>
            (a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    }
    return cudaErrorNotSupported;
}

// tensorA Element tensorC
template<typename Op>
static inline cudaError_t selectIndex(
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
        if (aDesc.order == cDesc.order && 
            aDesc.order == o0Desc.order && aDesc.order == o1Desc.order &&
            aDesc.isDense() && cDesc.isDense() &&
            o0Desc.isDense() && o1Desc.isDense()) {
            // if flattened, then cast to a packed element type if
            // possible to use simd instructions
            return flattened<typename Op::packed>
                (a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
        }
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
    case real32F:  return selectIndex<Op<A,B,C,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<A,B,C,float16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<A,B,C,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<A,B,C,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<A,B,C,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<A,B,C,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<A,B,C,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<A,B,C,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<A,B,C,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<A,B,C,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<A,B,C,complexf>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
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
    case real32F:  return selectIndex<Op<A,A,C,float>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<A,A,C,float16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<A,A,C,bfloat16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<A,A,C,double>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<A,A,C,int32_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<A,A,C,uint8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<A,A,C,int8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<A,A,C,uint16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<A,A,C,int16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<A,A,C,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<A,A,C,complexf>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
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
    case real32F:  return selectIndex<Op<float,float,float,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,float16,float16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bfloat16,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,double,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,int32_t,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,uint8_t,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,int8_t,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,uint16_t,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,int16_t,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,complexf,complexf>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
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
    assert(aDesc.type == oDesc.type);

    switch(aDesc.type) {
    case real32F:  return selectIndex<Op<float,float,float,float>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,float16,float16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bfloat16,bfloat16>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,double,double>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,int32_t,int32_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,uint8_t,uint8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,int8_t,int8_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,uint16_t,uint16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,int16_t,int16_t>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool,bool>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,complexf,complexf>>(a, aDesc, e, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
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
    case real32F:  return selectIndex<Op<float>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16F:  return selectIndex<Op<float16>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16BF: return selectIndex<Op<bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real64F:  return selectIndex<Op<double>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real32I:  return selectIndex<Op<int32_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8U:   return selectIndex<Op<uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8I:   return selectIndex<Op<int8_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16U:  return selectIndex<Op<uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16I:  return selectIndex<Op<int16_t>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case boolean:  return selectIndex<Op<bool>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case complex32F: return selectIndex<Op<complexf>>(a, aDesc, b, bDesc, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
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
    case real32F:  return selectIndex<Op<float>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16F:  return selectIndex<Op<float16>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16BF: return selectIndex<Op<bfloat16>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real64F:  return selectIndex<Op<double>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real32I:  return selectIndex<Op<int32_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8U:   return selectIndex<Op<uint8_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real8I:   return selectIndex<Op<int8_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16U:  return selectIndex<Op<uint16_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case real16I:  return selectIndex<Op<int16_t>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case boolean:  return selectIndex<Op<bool>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
    case complex32F: return selectIndex<Op<complexf>>(a, aDesc, b, c, cDesc, out0, o0Desc, out1, o1Desc, stream);
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
    case real16F:  return selectOut<Op, float16,float16,float16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectOut<Op, bfloat16,bfloat16,bfloat16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectOut<Op, double,double,double>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t,int32_t,int32_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uint8_t,uint8_t,uint8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectOut<Op, int8_t,int8_t,int8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectOut<Op, uint16_t,uint16_t,uint16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectOut<Op, int16_t,int16_t,int16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectOut<Op, bool,bool,bool>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectOut<Op, complexf,complexf,complexf>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
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
    case real32F:  return selectIndex<Op<float,float,bool,float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectIndex<Op<float16,float16,bool,float16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectIndex<Op<bfloat16,bfloat16,bool,bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectIndex<Op<double,double,bool,double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectIndex<Op<int32_t,int32_t,bool,int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectIndex<Op<uint8_t,uint8_t,bool,uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectIndex<Op<int8_t,int8_t,bool,int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectIndex<Op<uint16_t,uint16_t,bool,uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectIndex<Op<int16_t,int16_t,bool,int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectIndex<Op<bool,bool,bool,bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectIndex<Op<complexf,complexf,bool,complexf>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
