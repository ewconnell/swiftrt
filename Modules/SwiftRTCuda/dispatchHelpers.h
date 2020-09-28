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
// #ifndef dispatchHelpers_h
// #define dispatchHelpers_h
#ifndef disp_h
#define disp_h

#include <stdint.h>
#include <cuda.h>
#include <type_traits>
#include "complex.h"
#include "index.h"
#include "srttypes.h"

//==============================================================================
// conformance helpers
template<typename A, typename O>
inline constexpr bool isSame() {
    return std::is_same<A,O>::value;
}

template<typename A>
inline constexpr bool isInteger() {
    return std::is_integral<A>::value;
}

template<typename A>
inline constexpr bool isFloating() {
    return std::is_floating_point<A>::value;
}

template<typename A>
inline constexpr bool isComplex() {
    return std::is_same<A, Complex<float>>::value;
}

template<typename A>
inline constexpr bool isBool() {
    return std::is_same<A,bool>::value;
}

template<typename A>
inline constexpr bool isNumeric() {
    return isInteger<A>() || isFloating<A>() || isComplex<A>();
}

template<typename A>
inline constexpr bool isSignedNumeric() {
    return isNumeric<A>() && std::is_signed<A>::value;
}

//==============================================================================
// operator macros
#define OpT(OpName, name, conformance) \
template<typename T, typename O> struct OpName { \
    typedef T A; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (conforms()) out = name(a); \
    } \
};

#define OpTT(OpName, name, conformance) \
template<typename T, typename O> struct OpName { \
    typedef T A; typedef T B; typedef O Out; \
    constexpr static bool conforms() { return (conformance); } \
    __device__ inline static void op(const A& a, const B& b, O& out) { \
        if constexpr (conforms()) out = name(a, b); \
    } \
};

//==============================================================================
// used for casting between gpu simd types and uint32_t
#define UINT_CREF(_v) reinterpret_cast<const unsigned&>(_v)
#define CAST(type, _v) (*reinterpret_cast<const type*>(&(_v)))

//==============================================================================
/// fillWord
/// packs elements of size T into a 32 bit word
template<typename T>
inline uint32_t fillWord(const void* pValue) {
    static_assert(sizeof(T) <= sizeof(uint32_t), "T cannot be larger than return type");
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    uint32_t value = uint32_t(*static_cast<const T*>(pValue));
    uint32_t out = value;
    #pragma unroll
    for (int i = 0; i < sizeof(uint32_t) / sizeof(T); ++i) {
        out = (out << sizeof(T) * 8) | value;
    }
    return out;
}

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// shiftDownRoundingUp
inline unsigned shiftDownRoundingUp(unsigned num, unsigned shift) {
    unsigned count = (num + (1 << shift) - 1) >> shift;
    return count;
}

//------------------------------------------------------------------------------
/// roundUp
// tiles should always be shaped as a power of 2
// TODO: there should be something faster than this
inline unsigned roundUp(unsigned n, unsigned multiple) {
    return (n + multiple - 1) / multiple;
}

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic
// tile selection
template<unsigned Rank>
inline dim3 tileSize(const TensorDescriptor& oDesc) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return oDesc.count >= 1024 ? dim3(1024) : dim3(32);
    if (Rank == 2) return dim3(16, 16);
    if (Rank == 3) return dim3(16, 8, 8);
}

inline dim3 tileSize(int count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

template<unsigned Rank>
inline dim3 gridSize(const TensorDescriptor& oDesc, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return (oDesc.count + tile.x - 1) / tile.x;

    if (Rank == 2) return dim3(roundUp(oDesc.shape[0], tile.y), 
                               roundUp(oDesc.shape[1], tile.x));
    
    if (Rank == 3) return dim3(roundUp(oDesc.shape[0], tile.z), 
                               roundUp(oDesc.shape[1], tile.y), 
                               roundUp(oDesc.shape[2], tile.x));
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
// tensorA
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

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// flattened tensorA
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapA<Op,Flat,Flat><<<grid, tile, 0, stream>>>(a, Flat(aDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
/// flattened tensorA
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        const typename Op::B  e = *static_cast<const typename Op::B*>(pElement);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAE<Op,Flat,Flat><<<grid, tile, 0, stream>>>(a, Flat(aDesc), e, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
/// flattened tensorA parameter value
template<typename Op, typename Scalar>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAScalar<Op,Scalar,Flat,Flat>
            <<<grid, tile, 0, stream>>>(a, Flat(aDesc), value, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
/// flattened tensorA
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        const typename Op::B* b = static_cast<const typename Op::B*>(pB);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
        dim3 tile = tileSize(packedCount);
        dim3 grid = gridSize<1>(oDesc, tile);

        mapAB<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), out, Flat(oDesc));
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
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        dim3 tile = tileSize<IndexO::Rank>(oDesc);
        dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

        mapA<Op,IndexA,IndexO>
            <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

// initIndex tensorA Element
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    const void* pElement,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        const typename Op::A e = *static_cast<const typename Op::A*>(pElement);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        dim3 tile = tileSize<IndexO::Rank>(oDesc);
        dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

        mapAScalar<Op,typename Op::A,IndexA,IndexO>
            <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), e, out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        const typename Op::B* b = static_cast<const typename Op::B*>(pB);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        dim3 tile = tileSize<IndexO::Rank>(oDesc);
        dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

        IndexA indexA = IndexA(aDesc);
        IndexB indexB = IndexB(bDesc);
        
        mapAB<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>
            (a, indexA, b, indexB, out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
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

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA Scalar
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, element, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, element, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, element, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA tensorB
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>,Strided<1>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>,Strided<2>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>,Strided<3>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select out
// converts from dynamic to static type and delegates for stride selection
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if (oDesc.isStrided()) {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, __half>>(a, aDesc, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, __nv_bfloat16>>(a, aDesc, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return flattened<Op<A, __half2>>(a, aDesc, out, oDesc, stream, 1);
        case real16BF: return flattened<Op<A, __nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case real64F:  return flattened<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return flattened<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return flattened<Op<A, uchar4>>(a, aDesc, out, oDesc, stream, 2);
        case real8I:   return flattened<Op<A, char4>>(a, aDesc, out, oDesc, stream, 2);
        case real16U:  return flattened<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
        case real16I:  return flattened<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
        default: return cudaErrorNotSupported;
        }
    }
}

// tensorA Element
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if (oDesc.isStrided()) {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, element, out, oDesc, stream);
        case real16F:  return selectRank<Op<A, __half>>(a, aDesc, element, out, oDesc, stream);
        case real16BF: return selectRank<Op<A, __nv_bfloat16>>(a, aDesc, element, out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, element, out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, element, out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, element, out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, element, out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, element, out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, element, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<A, float>>(a, aDesc, element, out, oDesc, stream);
        case real16F:  return flattened<Op<A, __half2>>(a, aDesc, element, out, oDesc, stream, 1);
        case real16BF: return flattened<Op<A, __nv_bfloat162>>(a, aDesc, element, out, oDesc, stream, 1);
        case real64F:  return flattened<Op<A, double>>(a, aDesc, element, out, oDesc, stream);
        case real32I:  return flattened<Op<A, int32_t>>(a, aDesc, element, out, oDesc, stream);
        case real8U:   return flattened<Op<A, uchar4>>(a, aDesc, element, out, oDesc, stream, 2);
        case real8I:   return flattened<Op<A, char4>>(a, aDesc, element, out, oDesc, stream, 2);
        case real16U:  return flattened<Op<A, short2>>(a, aDesc, element, out, oDesc, stream, 1);
        case real16I:  return flattened<Op<A, short2>>(a, aDesc, element, out, oDesc, stream, 1);
        default: return cudaErrorNotSupported;
        }
    }
}

// tensorA tensorB
template<template<typename A, typename O> class Op, typename A>
static cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if (oDesc.isStrided()) {
        switch(oDesc.type) {
        case real32F:  return selectRank<Op<A, float>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16F:  return selectRank<Op<A, __half>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16BF: return selectRank<Op<A, __nv_bfloat16>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real64F:  return selectRank<Op<A, double>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real32I:  return selectRank<Op<A, int32_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8I:   return selectRank<Op<A, int8_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8U:   return selectRank<Op<A, uint8_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16I:  return selectRank<Op<A, int16_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16U:  return selectRank<Op<A, uint16_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<A, float>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16F:  return flattened<Op<A, __half2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real16BF: return flattened<Op<A, __nv_bfloat162>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real64F:  return flattened<Op<A, double>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real32I:  return flattened<Op<A, int32_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8U:   return flattened<Op<A, uchar4>>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
        case real8I:   return flattened<Op<A, char4>>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
        case real16U:  return flattened<Op<A, short2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real16I:  return flattened<Op<A, short2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case boolean:  return flattened<Op<A, bool4>>(a, aDesc, b, bDesc, out, oDesc, stream, 2);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// select tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectOut<Op, __half2>(a, aDesc, out, oDesc, stream, 1);
    case real16BF: return selectOut<Op, __nv_bfloat162>(a, aDesc, out, oDesc, stream, 1);
    case real64F:  return selectOut<Op, double>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uchar4>(a, aDesc, out, oDesc, stream, 2);
    case real8I:   return selectOut<Op, char4>(a, aDesc, out, oDesc, stream, 2);
    case real16U:  return selectOut<Op, short2>(a, aDesc, out, oDesc, stream, 1);
    case real16I:  return selectOut<Op, short2>(a, aDesc, out, oDesc, stream, 1);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float>(a, aDesc, element, out, oDesc, stream);
    case real16F:  return selectOut<Op, __half2>(a, aDesc, element, out, oDesc, stream, 1);
    case real16BF: return selectOut<Op, __nv_bfloat162>(a, aDesc, element, out, oDesc, stream, 1);
    case real64F:  return selectOut<Op, double>(a, aDesc, element, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, element, out, oDesc, stream);
    case real8U:   return selectOut<Op, uchar4>(a, aDesc, element, out, oDesc, stream, 2);
    case real8I:   return selectOut<Op, char4>(a, aDesc, element, out, oDesc, stream, 2);
    case real16U:  return selectOut<Op, short2>(a, aDesc, element, out, oDesc, stream, 1);
    case real16I:  return selectOut<Op, short2>(a, aDesc, element, out, oDesc, stream, 1);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename A, typename O> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(aDesc.type) {
    case real32F:  return selectOut<Op, float>(a, aDesc, b, bDesc,out, oDesc, stream);
    case real16F:  return selectOut<Op, __half2>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
    case real16BF: return selectOut<Op, __nv_bfloat162>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
    case real64F:  return selectOut<Op, double>(a, aDesc, b, bDesc,out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, b, bDesc,out, oDesc, stream);
    case real8U:   return selectOut<Op, uchar4>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
    case real8I:   return selectOut<Op, char4>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
    case real16U:  return selectOut<Op, short2>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
    case real16I:  return selectOut<Op, short2>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
    case boolean:  return selectOut<Op, bool4>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
    default: return cudaErrorNotSupported;
    }
}

#endif