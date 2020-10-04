//******************************************************************************
// Copyright 2020 Google LLC
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
#include <type_traits>
#include "srttypes.h"
#include "complex.h"
#include "index.h"

//==============================================================================
// conformance helpers
template<typename A, typename O>
inline constexpr bool isSame() {
    return std::is_same<A,O>::value;
}

template<typename A>
inline constexpr bool isInteger() {
    return std::is_integral<A>::value ||
        std::is_same<A,char4>::value  || std::is_same<A,uchar4>::value ||
        std::is_same<A,short2>::value || std::is_same<A,ushort2>::value;
}

template<typename A>
inline constexpr bool isFloating() {
    return 
        std::is_floating_point<A>::value ||
        std::is_same<A,float16>::value  || std::is_same<A,float162>::value ||
        std::is_same<A,bfloat16>::value || std::is_same<A,bfloat162>::value;
}

template<typename A>
inline constexpr bool isComplex() {
    return std::is_same<A, Complex<float>>::value;
}

template<typename A>
inline constexpr bool isBool() {
    return std::is_same<A,bool>::value ||
        std::is_same<A,bool2>::value || std::is_same<A,bool4>::value;
}

template<typename A>
inline constexpr bool isNumeric() {
    return isInteger<A>() || isFloating<A>() || isComplex<A>();
}

template<typename A>
inline constexpr bool isComparable() {
    return isNumeric<A>();
}

template<typename A>
inline constexpr bool isEquatable() {
    return isNumeric<A>() || isBool<A>();
}

template<typename A>
inline constexpr bool isSignedNumeric() {
    return isNumeric<A>() && std::is_signed<A>::value;
}

template<typename A>
inline constexpr bool isPacked() {
    return 
    std::is_same<A,bool2>::value  || std::is_same<A,bool4>::value ||
    std::is_same<A,char4>::value  || std::is_same<A,uchar4>::value ||
    std::is_same<A,short2>::value || std::is_same<A,ushort2>::value ||
    std::is_same<A,float162>::value || std::is_same<A,bfloat162>::value;
}

//==============================================================================
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename T> struct packed {
    typedef T type;
    inline static T value(const T v) { return v; }
};

template<> struct packed<int8_t> {
    typedef char4 type;
    inline static type value(const int8_t v) {
        type p; p.x = v; p.y = v; p.z = v; p.w = v; return p;
    }
};

template<> struct packed<uint8_t> {
    typedef uchar4 type;
    inline static type value(const uint8_t v) {
        type p; p.x = v; p.y = v; p.z = v; p.w = v; return p;
    }
};

template<> struct packed<int16_t> {
    typedef short2 type;
    inline static type value(const int16_t v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<uint16_t> {
    typedef ushort2 type;
    inline static type value(const uint16_t v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<float16> {
    typedef float162 type;
    inline static type value(const float16 v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<bfloat16> {
    typedef bfloat162 type;
    inline static type value(const bfloat16 v) {
        type p; p.x = v; p.y = v; return p;
    }
};

//--------------------------------------
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename A, typename O>
struct matching_packed { typedef O type; };
template<> struct matching_packed<char4, bool> { typedef bool4 type; };
template<> struct matching_packed<char4, int8_t> { typedef char4 type; };

template<> struct matching_packed<uchar4, bool> { typedef bool4 type; };
template<> struct matching_packed<uchar4, uint8_t> { typedef uchar4 type; };

template<> struct matching_packed<short2, bool> { typedef bool2 type; };
template<> struct matching_packed<short2, int16_t> { typedef short2 type; };

template<> struct matching_packed<ushort2, bool> { typedef bool2 type; };
template<> struct matching_packed<ushort2, uint16_t> { typedef ushort2 type; };

template<> struct matching_packed<float162, bool> { typedef bool2 type; };
template<> struct matching_packed<float162, float16> { typedef float162 type; };

template<> struct matching_packed<bfloat162, bool> { typedef bool2 type; };
template<> struct matching_packed<bfloat162, bfloat16> { typedef bfloat162 type; };

//--------------------------------------
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename A> struct packing {static const int count = 1; };
template<> struct packing<char4> { static const int count = 4; };
template<> struct packing<const char4> { static const int count = 4; };
template<> struct packing<uchar4> { static const int count = 4; };
template<> struct packing<const uchar4> { static const int count = 4; };
template<> struct packing<bool4> { static const int count = 4; };
template<> struct packing<const bool4> { static const int count = 4; };

template<> struct packing<bool2> { static const int count = 2; };
template<> struct packing<const bool2> { static const int count = 2; };
template<> struct packing<short2> { static const int count = 2; };
template<> struct packing<const short2> { static const int count = 2; };
template<> struct packing<ushort2> { static const int count = 2; };
template<> struct packing<const ushort2> { static const int count = 2; };
template<> struct packing<float162> { static const int count = 2; };
template<> struct packing<const float162> { static const int count = 2; };
template<> struct packing<bfloat162> { static const int count = 2; };
template<> struct packing<const bfloat162> { static const int count = 2; };

//==============================================================================
// operator macros
// `packed` is a version of the operator where types smaller than 32 bit
// are retyped into packed versions to use with gpu SIMD instructions
#define Op1(OpName, name, conformance) \
template<typename T, typename O> struct OpName { \
    static_assert(isPacked<T>() == isPacked<O>(), "packed type mismatch"); \
    typedef T A; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (conforms()) out = name(a); \
    } \
    typedef typename packed<T>::type PT; \
    typedef typename matching_packed<PT,O>::type PO; \
    typedef OpName<PT,PO> packed; \
};

#define Op2(OpName, name, conformance) \
template<typename T, typename O> struct OpName { \
    static_assert(isPacked<T>() == isPacked<O>(), "packed type mismatch"); \
    typedef T A; typedef T B; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, const B& b, O& out) { \
        if constexpr (conforms()) out = name(a, b); \
    } \
    typedef typename packed<T>::type PT; \
    typedef typename matching_packed<PT,O>::type PO; \
    typedef OpName<PT,PO> packed; \
};

#define Op3(OpName, name, conformance) \
template<typename T, typename O> struct OpName { \
    static_assert(isPacked<T>() == isPacked<O>(), "packed type mismatch"); \
    typedef T A; typedef T B; typedef T C; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, const B& b, const C& c, O& out) { \
        if constexpr (conforms()) out = name(a, b, c); \
    } \
    typedef typename packed<T>::type PT; \
    typedef typename matching_packed<PT,O>::type PO; \
    typedef OpName<PT,PO> packed; \
};

#define OpTTU(OpName, name, conformance) \
template<typename T, typename U, typename O> struct OpName { \
    static_assert(isPacked<T>() == isPacked<U>() && \
                  isPacked<T>() == isPacked<O>(), "packed type mismatch"); \
    typedef T A; typedef T B; typedef U C; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, const B& b, const C& c, O& out) { \
        if constexpr (conforms()) out = name(a, b, c); \
    } \
    typedef typename packed<T>::type PT; \
    typedef typename matching_packed<PT,U>::type PU; \
    typedef typename matching_packed<PT,O>::type PO; \
    typedef OpName<PT,PU,PO> packed; \
};

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// divideRoundingUp
inline int divideRoundingUp(int num, int divisor) {
    return (num + divisor - 1) / divisor;
}

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic

//--------------------------------------
// tile selection
template<unsigned Rank>
inline dim3 tileSize(const TensorDescriptor& oDesc) {
    static_assert(Rank <= 3, "not implemented");
}

template<> inline dim3 tileSize<1>(const TensorDescriptor& oDesc) {
    return oDesc.count >= 1024 ? dim3(1024) : dim3(32);
}

template<> inline dim3 tileSize<2>(const TensorDescriptor& oDesc) {
    return dim3(16, 16);
}

template<> inline dim3 tileSize<3>(const TensorDescriptor& oDesc) {
    return dim3(16, 8, 8);
}

inline dim3 tileSize(int count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

//--------------------------------------
// grid selection
template<unsigned Rank>
inline dim3 gridSize(const TensorDescriptor& oDesc, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
}

template<>
inline dim3 gridSize<1>(const TensorDescriptor& oDesc, const dim3& tile) {
    return (oDesc.count + tile.x - 1) / tile.x;
}

template<>
inline dim3 gridSize<2>(const TensorDescriptor& oDesc, const dim3& tile) {
    return dim3(divideRoundingUp(oDesc.shape[0], tile.y), 
                divideRoundingUp(oDesc.shape[1], tile.x));
}

template<>
inline dim3 gridSize<3>(const TensorDescriptor& oDesc, const dim3& tile) {
    return dim3(divideRoundingUp(oDesc.shape[0], tile.z), 
                divideRoundingUp(oDesc.shape[1], tile.y), 
                divideRoundingUp(oDesc.shape[2], tile.x));
}

//==============================================================================
// kernels
//==============================================================================

//------------------------------------------------------------------------------
// tensorA
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

//------------------------------------------------------------------------------
// Element tensorA
// used when op can't reverse order like divide and subtract
template<typename Op, typename IndexA, typename IndexO>
__global__ void mapEA(
    const typename Op::A  element,
    const typename Op::A* __restrict__ a, const IndexA indexA,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int io = indexO.linear(position);
        Op::op(element, a[ia], out[io]);
    }
}

//------------------------------------------------------------------------------
// tensorA Scalar
template<typename Op, typename Scalar, typename IndexA, typename IndexO>
__global__ void mapAScalar(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    Scalar value,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        Op::op(a[ia], value, out[io]);
    }
}

//------------------------------------------------------------------------------
// tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
__global__ void mapAB(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        Op::op(a[ia], b[ib], out[io]);
    }
}

//------------------------------------------------------------------------------
// tensorA tensorB tensorC
template<typename Op, typename IndexA, typename IndexB,
         typename IndexC, typename IndexO>
__global__ void mapABC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int ic = indexC.linear(position);
        int io = indexO.linear(position);
        Op::op(a[ia], b[ib], c[ic], out[io]);
    }
}

//------------------------------------------------------------------------------
// tensorA tensorB Element
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
__global__ void mapABE(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    const typename Op::A  element,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        Op::op(a[ia], b[ib], element, out[io]);
    }
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//==============================================================================
/// flattened

//--------------------------------------
// tensorA
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
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

//--------------------------------------
// tensorA Element
template<typename Op>
static cudaError_t flattened(
    const void* pElement,
    const void* pA, const TensorDescriptor& aDesc,
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

        mapEA<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (e, a, Flat(aDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA parameter value
template<typename Op, typename P>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc, 
    const P param,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAScalar<Op,P,Flat,Flat>
            <<<grid, tile, 0, stream>>>(a, Flat(aDesc), param, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

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

        mapABC<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), c, Flat(cDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// tensorA tensorB Element
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        using A = const typename Op::A;
        using B = const typename Op::B;
        using E = const typename Op::C;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        B* b = static_cast<B*>(pB);
        E  e = *static_cast<E*>(pElement);
        Out* out = static_cast<Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = divideRoundingUp(oDesc.count, packing<A>::count);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapABE<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), e, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex tensorA
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
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

    mapA<Op,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc),
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA Element
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

    mapAScalar<Op,typename Op::A,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        e, 
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA Element
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pElement,
    const void* pA, const TensorDescriptor& aDesc, 
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

    mapEA<Op,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        e, 
        a, IndexA(aDesc),
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA Parameter
template<typename Op, typename IndexA, typename IndexO, typename P>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    const P param,
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

    mapAScalar<Op,P,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        param, 
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA tensorB
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

    IndexA indexA = IndexA(aDesc);
    IndexB indexB = IndexB(bDesc);
    
    mapAB<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>(
        a, indexA, 
        b, indexB,
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    const void* pE,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using B = const typename Op::B;
    using E = A;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    B* b = static_cast<B*>(pB);
    E  e = *static_cast<B*>(pE);
    Out* out = static_cast<Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapABE<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA(aDesc), 
        b, IndexB(bDesc),
        e,
        out, IndexO(oDesc));
    return cudaSuccess;
}

//--------------------------------------
// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB,
         typename IndexC, typename IndexO>
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

//==============================================================================
// selectRank tensorA
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(a, aDesc, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<4>>(a, aDesc, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<5>>(a, aDesc, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<6>>(a, aDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank tensorA Element
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(a, aDesc, element, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, element, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, element, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, element, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<4>>(a, aDesc, element, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<5>>(a, aDesc, element, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<6>>(a, aDesc, element, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank Element tensorA
template<typename Op>
static cudaError_t selectRank(
    const void* element,
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(element, a, aDesc, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>>(element, a, aDesc, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>>(element, a, aDesc, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>>(element, a, aDesc, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<4>>(element, a, aDesc, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<5>>(element, a, aDesc, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<6>>(element, a, aDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank tensorA Element
template<typename Op, typename P>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const P param,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed, P>(a, aDesc, param, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>, P>(a, aDesc, param, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>, P>(a, aDesc, param, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>, P>(a, aDesc, param, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<4>, P>(a, aDesc, param, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<5>, P>(a, aDesc, param, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<6>, P>(a, aDesc, param, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank tensorA tensorB
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(a, aDesc, b, bDesc, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>,Strided<1>>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>,Strided<2>>(a, aDesc, b, bDesc, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>,Strided<3>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<1>,Strided<4>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<2>,Strided<5>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<3>,Strided<6>>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank tensorA tensorB Element
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>,Strided<1>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>,Strided<2>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>,Strided<3>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<1>,Strided<4>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<2>,Strided<5>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<3>,Strided<6>>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//--------------------------------------
// selectRank tensorA tensorB tensorC
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == cDesc.rank &&
           aDesc.rank == oDesc.rank);
    if constexpr (Op::conforms()) {
        if (aDesc.isDense() && cDesc.isDense() && oDesc.isDense()) {
            return flattened<typename Op::packed>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        } else {
            switch (oDesc.rank) {
            case 1: return initIndex<Op,Strided<1>,Strided<1>,Strided<1>,Strided<1>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
            case 2: return initIndex<Op,Strided<2>,Strided<2>,Strided<2>,Strided<2>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
            case 3: return initIndex<Op,Strided<3>,Strided<3>,Strided<3>,Strided<3>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
            // case 4: return initIndex<Op,Strided<4>,Strided<1>,Strided<4>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 5: return initIndex<Op,Strided<5>,Strided<2>,Strided<5>>(a, aDesc, b, bDesc, out, oDesc, stream);
            // case 6: return initIndex<Op,Strided<6>,Strided<3>,Strided<6>>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// select out
// converts from dynamic to static type and delegates for stride selection

// selectOut tensorA
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        return selectRank<Op<A,A>>(a, aDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, float16>>(a, aDesc, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, bfloat16>>(a, aDesc, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, out, oDesc, stream);
        case boolean:  return selectRank<Op<A, bool>>(a, aDesc, out, oDesc, stream);
        case complex32F:  return selectRank<Op<A, Complex<float>>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//--------------------------------------
// tensorA Element
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        return selectRank<Op<A,A>>(a, aDesc, element, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, element, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, float16>>(a, aDesc, element, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, bfloat16>>(a, aDesc, element, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, element, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, element, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, element, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, element, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, element, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, element, out, oDesc, stream);
        case boolean:  return selectRank<Op<A, bool>>(a, aDesc, element, out, oDesc, stream);
        case complex32F:  return selectRank<Op<A, Complex<float>>>(a, aDesc, element, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//--------------------------------------
// Element tensorA
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* element,
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        return selectRank<Op<A,A>>(element, a, aDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(element, a, aDesc, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, float16>>(element, a, aDesc, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, bfloat16>>(element, a, aDesc, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(element, a, aDesc, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(element, a, aDesc, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(element, a, aDesc, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(element, a, aDesc, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(element, a, aDesc, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(element, a, aDesc, out, oDesc, stream);
        case boolean:  return selectRank<Op<A, bool>>(element, a, aDesc, out, oDesc, stream);
        case complex32F:  return selectRank<Op<A, Complex<float>>>(element, a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//--------------------------------------
// tensorA tensorB
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        return selectRank<Op<A,A>>(a, aDesc, b, bDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, float16>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case boolean:  return selectRank<Op<A, bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case complex32F:  return selectRank<Op<A, Complex<float>>>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//--------------------------------------
// tensorA tensorB tensorC
template<template<typename A, typename C, typename O> class Op, typename A, typename C>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.type == oDesc.type) {
        return selectRank<Op<A,C,A>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, C, float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, C, float16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, C, bfloat16>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, C, double>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, C, int32_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, C, int8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, C, uint8_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, C, int16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, C, uint16_t>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case boolean:  return selectRank<Op<A, C, bool>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case complex32F:  return selectRank<Op<A, C, Complex<float>>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// select
// converts from dynamic to static type and delegates for stride selection

//--------------------------------------
// tensorA
// selectOut is called for the A to O typecast case
template<template<typename A, typename O> class Op>
static cudaError_t select(
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
    case complex32F: return selectOut<Op, Complex<float>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Element
// selectRank is called because current cases A == O
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectRank<Op<float,float>>(a, aDesc, element, out, oDesc, stream);
    case real16F:  return selectRank<Op<float16,float16>>(a, aDesc, element, out, oDesc, stream);
    case real16BF: return selectRank<Op<bfloat16,bfloat16>>(a, aDesc, element, out, oDesc, stream);
    case real64F:  return selectRank<Op<double,double>>(a, aDesc, element, out, oDesc, stream);
    case real32I:  return selectRank<Op<int32_t, int32_t>>(a, aDesc, element, out, oDesc, stream);
    case real8U:   return selectRank<Op<uint8_t,uint8_t>>(a, aDesc, element, out, oDesc, stream);
    case real8I:   return selectRank<Op<int8_t,int8_t>>(a, aDesc, element, out, oDesc, stream);
    case real16U:  return selectRank<Op<uint16_t,uint16_t>>(a, aDesc, element, out, oDesc, stream);
    case real16I:  return selectRank<Op<int16_t,int16_t>>(a, aDesc, element, out, oDesc, stream);
    case complex32F: return selectRank<Op<Complex<float>,Complex<float>>>(a, aDesc, element, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// Element tensorA
// selectRank is called because current cases A == O
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* element,
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectRank<Op<float,float>>(element, a, aDesc, out, oDesc, stream);
    case real16F:  return selectRank<Op<float16,float16>>(element, a, aDesc, out, oDesc, stream);
    case real16BF: return selectRank<Op<bfloat16,bfloat16>>(element, a, aDesc, out, oDesc, stream);
    case real64F:  return selectRank<Op<double,double>>(element, a, aDesc, out, oDesc, stream);
    case real32I:  return selectRank<Op<int32_t,int32_t>>(element, a, aDesc, out, oDesc, stream);
    case real8U:   return selectRank<Op<uint8_t,uint8_t>>(element, a, aDesc, out, oDesc, stream);
    case real8I:   return selectRank<Op<int8_t,int8_t>>(element, a, aDesc, out, oDesc, stream);
    case real16U:  return selectRank<Op<uint16_t,uint16_t>>(element, a, aDesc, out, oDesc, stream);
    case real16I:  return selectRank<Op<int16_t,int16_t>>(element, a, aDesc, out, oDesc, stream);
    case complex32F: return selectRank<Op<Complex<float>,Complex<float>>>(element, a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// tensorA Parameter
// selectRank is called because current cases A == O
template<template<typename A, typename O> class Op, typename P>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const P param,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef Complex<float> CF;

    switch(aDesc.type) {
    case real32F:  return selectRank<Op<float,float>, P>(a, aDesc, param, out, oDesc, stream);
    case real16F:  return selectRank<Op<float16,float16>, P>(a, aDesc, param, out, oDesc, stream);
    case real16BF: return selectRank<Op<bfloat16,bfloat16>, P>(a, aDesc, param, out, oDesc, stream);
    case real64F:  return selectRank<Op<double,double>, P>(a, aDesc, param, out, oDesc, stream);
    case real32I:  return selectRank<Op<int32_t, int32_t>, P>(a, aDesc, param, out, oDesc, stream);
    case real8U:   return selectRank<Op<uint8_t,uint8_t>, P>(a, aDesc, param, out, oDesc, stream);
    case real8I:   return selectRank<Op<int8_t,int8_t>, P>(a, aDesc, param, out, oDesc, stream);
    case real16U:  return selectRank<Op<uint16_t,uint16_t>, P>(a, aDesc, param, out, oDesc, stream);
    case real16I:  return selectRank<Op<int16_t,int16_t>, P>(a, aDesc, param, out, oDesc, stream);
    case complex32F: return selectRank<Op<CF,CF>, P>(a, aDesc, param, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// select tensorA tensorB
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type);
    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectOut<Op, float16>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectOut<Op, bfloat16>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectOut<Op, double>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uint8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real8I:   return selectOut<Op, int8_t>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16U:  return selectOut<Op, uint16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16I:  return selectOut<Op, int16_t>(a, aDesc, b, bDesc, out, oDesc, stream);
    case boolean:  return selectOut<Op, bool>(a, aDesc, b, bDesc, out, oDesc, stream);
    case complex32F:  return selectOut<Op, Complex<float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//--------------------------------------
// select tensorA tensorB Element
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type);
    if (aDesc.type == oDesc.type) {
        switch(aDesc.type) {
        case real32F:  return selectRank<Op<float,float>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16F:  return selectRank<Op<float16, float16>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16BF: return selectRank<Op<bfloat16, bfloat16>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real64F:  return selectRank<Op<double, double>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real32I:  return selectRank<Op<int32_t, int32_t>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real8U:   return selectRank<Op<uint8_t, uint8_t>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real8I:   return selectRank<Op<int8_t, int8_t>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16U:  return selectRank<Op<uint16_t, uint16_t>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16I:  return selectRank<Op<int16_t, int16_t>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case boolean:  return selectRank<Op<bool, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case complex32F:  return selectRank<Op<Complex<float>, Complex<float>>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        // the only other possiblity for this pattern right now
        assert(oDesc.type == boolean);
        switch(aDesc.type) {
        case real32F:  return selectRank<Op<float,bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16F:  return selectRank<Op<float16, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16BF: return selectRank<Op<bfloat16, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real64F:  return selectRank<Op<double, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real32I:  return selectRank<Op<int32_t, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real8U:   return selectRank<Op<uint8_t, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real8I:   return selectRank<Op<int8_t, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16U:  return selectRank<Op<uint16_t, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case real16I:  return selectRank<Op<int16_t, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case boolean:  return selectRank<Op<bool, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        case complex32F: return selectRank<Op<Complex<float>, bool>>(a, aDesc, b, bDesc, element, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// select tensorA tensorB tensorC
template<template<typename A, typename C, typename O> class Op, typename A>
static cudaError_t selectC(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type);
    if (aDesc.type == cDesc.type) {
        return selectOut<Op,A,A>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    } else {
        switch(cDesc.type) {
        case real32F:  return selectOut<Op, A, float>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16F:  return selectOut<Op, A, float16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16BF: return selectOut<Op, A, bfloat16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real64F:  return selectOut<Op, A, double>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real32I:  return selectOut<Op, A, int32_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real8U:   return selectOut<Op, A, uint8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real8I:   return selectOut<Op, A, int8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16U:  return selectOut<Op, A, uint16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case real16I:  return selectOut<Op, A, int16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case boolean:  return selectOut<Op, A, bool>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        case complex32F:  return selectOut<Op, A, Complex<float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// select tensorA tensorB tensorC
template<template<typename A, typename C, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type);
    switch(aDesc.type) {
    case real32F:  return selectC<Op, float>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16F:  return selectC<Op, float16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16BF: return selectC<Op, bfloat16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real64F:  return selectC<Op, double>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real32I:  return selectC<Op, int32_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8U:   return selectC<Op, uint8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real8I:   return selectC<Op, int8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16U:  return selectC<Op, uint16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case real16I:  return selectC<Op, int16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case boolean:  return selectC<Op, bool>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    case complex32F: return selectC<Op, Complex<float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
