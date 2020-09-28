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
#include "types.h"

//==============================================================================
// conformance helpers
template<typename A, typename O>
inline constexpr auto isSame() {
    return std::is_same<A,O>::value;
}

template<typename A>
inline constexpr auto isInteger() {
    return std::is_integral<A>::value;
}

template<typename A>
inline constexpr auto isFloating() {
    return std::is_floating_point<A>::value;
}

template<typename A>
inline constexpr auto isComplex() {
    return std::is_same<A, Complex<float>>::value;
}

template<typename A>
inline constexpr auto isBool() {
    return std::is_same<A,bool>::value;
}

template<typename A>
inline constexpr auto isNumeric() {
    return isInteger<A>() || isFloating<A>() || isComplex<A>();
}

template<typename A>
inline constexpr auto isSignedNumeric() {
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
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// shiftDownRoundingUp
inline unsigned shiftDownRoundingUpNew(unsigned num, unsigned shift) {
    unsigned count = (num + (1 << shift) - 1) >> shift;
    return count;
}

//------------------------------------------------------------------------------
/// roundUp
// tiles should always be shaped as a power of 2
// TODO: there should be something faster than this
inline unsigned roundUpNew(unsigned n, unsigned multiple) {
    return (n + multiple - 1) / multiple;
}

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic
// tile selection
template<unsigned Rank>
inline dim3 tileSizeNew(const TensorDescriptor& oDesc) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return oDesc.count >= 1024 ? dim3(1024) : dim3(32);
    if (Rank == 2) return dim3(16, 16);
    if (Rank == 3) return dim3(16, 8, 8);
}

inline dim3 tileSizeNew(int count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

template<unsigned Rank>
inline dim3 gridSizeNew(const TensorDescriptor& oDesc, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return (oDesc.count + tile.x - 1) / tile.x;

    if (Rank == 2) return dim3(roundUpNew(oDesc.shape[0], tile.y), 
                               roundUpNew(oDesc.shape[1], tile.x));
    
    if (Rank == 3) return dim3(roundUpNew(oDesc.shape[0], tile.z), 
                               roundUpNew(oDesc.shape[1], tile.y), 
                               roundUpNew(oDesc.shape[2], tile.x));
}

//==============================================================================
// kernels
//==============================================================================

//------------------------------------------------------------------------------
// tensorA
template<typename Op, typename IndexA, typename IndexO>
__global__ void mapANew(
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
// tensorA Scalar
template<typename Op, typename Scalar, typename IndexA, typename IndexO>
__global__ void mapAScalarNew(
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
__global__ void mapABNew(
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
static cudaError_t flattenedNew(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUpNew(oDesc.count, shiftCount);
        dim3 tile = tileSizeNew(packedCount);
        dim3 grid = gridSizeNew<1>(oDesc, tile);

        mapANew<Op,Flat,Flat><<<grid, tile, 0, stream>>>(a, Flat(aDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
/// flattened tensorA Scalar
template<typename Op, typename Scalar>
static cudaError_t flattenedNew(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    if constexpr (Op::conforms()) {
        const typename Op::In* a = static_cast<const typename Op::In*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        int packedCount = shiftDownRoundingUpNew(oDesc.count, shiftCount);
        dim3 tile = tileSizeNew(packedCount);
        dim3 grid = gridSizeNew<1>(oDesc, tile);

        mapAScalarNew<Op,Scalar,Flat,Flat>
            <<<grid, tile, 0, stream>>>(a, Flat(aDesc), value, out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
/// flattened tensorA
template<typename Op>
static cudaError_t flattenedNew(
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
        int packedCount = shiftDownRoundingUpNew(oDesc.count, shiftCount);
        dim3 tile = tileSizeNew(packedCount);
        dim3 grid = gridSizeNew<1>(oDesc, tile);

        mapABNew<Op,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), out, Flat(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex tensorA
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndexNew(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        const typename Op::A* a = static_cast<const typename Op::A*>(pA);
        typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

        // get tile and grid size for launch
        dim3 tile = tileSizeNew<IndexO::Rank>(oDesc);
        dim3 grid = gridSizeNew<IndexO::Rank>(oDesc, tile);

        mapANew<Op,IndexA,IndexO>
            <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

// initIndex tensorA Element
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndexNew(
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
        dim3 tile = tileSizeNew<IndexO::Rank>(oDesc);
        dim3 grid = gridSizeNew<IndexO::Rank>(oDesc, tile);

        mapAScalarNew<Op,typename Op::A,IndexA,IndexO>
            <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), e, out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndexNew(
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
        dim3 tile = tileSizeNew<IndexO::Rank>(oDesc);
        dim3 grid = gridSizeNew<IndexO::Rank>(oDesc, tile);

        IndexA indexA = IndexA(aDesc);
        IndexB indexB = IndexB(bDesc);
        
        mapABNew<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>
            (a, indexA, b, indexB, out, IndexO(oDesc));
        return cudaSuccess;
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// selectRank tensorA
template<typename Op>
static cudaError_t selectRankNew(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndexNew<Op,Strided<1>,Strided<1>>(a, aDesc, out, oDesc, stream);
    case 2: return initIndexNew<Op,Strided<2>,Strided<2>>(a, aDesc, out, oDesc, stream);
    case 3: return initIndexNew<Op,Strided<3>,Strided<3>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA Scalar
template<typename Op>
static cudaError_t selectRankNew(
    const void* a, const TensorDescriptor& aDesc,
    const void* element,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndexNew<Op,Strided<1>,Strided<1>>(a, aDesc, element, out, oDesc, stream);
    case 2: return initIndexNew<Op,Strided<2>,Strided<2>>(a, aDesc, element, out, oDesc, stream);
    case 3: return initIndexNew<Op,Strided<3>,Strided<3>>(a, aDesc, element, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA tensorB
template<typename Op>
static cudaError_t selectRankNew(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndexNew<Op,Strided<1>,Strided<1>,Strided<1>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 2: return initIndexNew<Op,Strided<2>,Strided<2>,Strided<2>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 3: return initIndexNew<Op,Strided<3>,Strided<3>,Strided<3>>
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
        case real32F:  return selectRankNew<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return selectRankNew<Op<A, __half>>(a, aDesc, out, oDesc, stream);
        case real16BF: return selectRankNew<Op<A, __nv_bfloat16>>(a, aDesc, out, oDesc, stream);
        case real64F:  return selectRankNew<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return selectRankNew<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8I:   return selectRankNew<Op<A, int8_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return selectRankNew<Op<A, uint8_t>>(a, aDesc, out, oDesc, stream);
        case real16I:  return selectRankNew<Op<A, int16_t>>(a, aDesc, out, oDesc, stream);
        case real16U:  return selectRankNew<Op<A, uint16_t>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattenedNew<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return flattenedNew<Op<A, __half2>>(a, aDesc, out, oDesc, stream, 1);
        case real16BF: return flattenedNew<Op<A, __nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case real64F:  return flattenedNew<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return flattenedNew<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return flattenedNew<Op<A, uchar4>>(a, aDesc, out, oDesc, stream, 2);
        case real8I:   return flattenedNew<Op<A, char4>>(a, aDesc, out, oDesc, stream, 2);
        case real16U:  return flattenedNew<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
        case real16I:  return flattenedNew<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
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
        case real32F:  return selectRankNew<Op<A, float>>(a, aDesc, element, out, oDesc, stream);
        case real16F:  return selectRankNew<Op<A, __half>>(a, aDesc, element, out, oDesc, stream);
        case real16BF: return selectRankNew<Op<A, __nv_bfloat16>>(a, aDesc, element, out, oDesc, stream);
        case real64F:  return selectRankNew<Op<A, double>>(a, aDesc, element, out, oDesc, stream);
        case real32I:  return selectRankNew<Op<A, int32_t>>(a, aDesc, element, out, oDesc, stream);
        case real8I:   return selectRankNew<Op<A, int8_t>>(a, aDesc, element, out, oDesc, stream);
        case real8U:   return selectRankNew<Op<A, uint8_t>>(a, aDesc, element, out, oDesc, stream);
        case real16I:  return selectRankNew<Op<A, int16_t>>(a, aDesc, element, out, oDesc, stream);
        case real16U:  return selectRankNew<Op<A, uint16_t>>(a, aDesc, element, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattenedNew<Op<A, float>>(a, aDesc, element, out, oDesc, stream);
        case real16F:  return flattenedNew<Op<A, __half2>>(a, aDesc, element, out, oDesc, stream, 1);
        case real16BF: return flattenedNew<Op<A, __nv_bfloat162>>(a, aDesc, element, out, oDesc, stream, 1);
        case real64F:  return flattenedNew<Op<A, double>>(a, aDesc, element, out, oDesc, stream);
        case real32I:  return flattenedNew<Op<A, int32_t>>(a, aDesc, element, out, oDesc, stream);
        case real8U:   return flattenedNew<Op<A, uchar4>>(a, aDesc, element, out, oDesc, stream, 2);
        case real8I:   return flattenedNew<Op<A, char4>>(a, aDesc, element, out, oDesc, stream, 2);
        case real16U:  return flattenedNew<Op<A, short2>>(a, aDesc, element, out, oDesc, stream, 1);
        case real16I:  return flattenedNew<Op<A, short2>>(a, aDesc, element, out, oDesc, stream, 1);
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
        case real32F:  return selectRankNew<Op<A, float>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16F:  return selectRankNew<Op<A, __half>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16BF: return selectRankNew<Op<A, __nv_bfloat16>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real64F:  return selectRankNew<Op<A, double>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real32I:  return selectRankNew<Op<A, int32_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8I:   return selectRankNew<Op<A, int8_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8U:   return selectRankNew<Op<A, uint8_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16I:  return selectRankNew<Op<A, int16_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16U:  return selectRankNew<Op<A, uint16_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case real32F:  return flattenedNew<Op<A, float>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real16F:  return flattenedNew<Op<A, __half2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real16BF: return flattenedNew<Op<A, __nv_bfloat162>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real64F:  return flattenedNew<Op<A, double>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real32I:  return flattenedNew<Op<A, int32_t>>(a, aDesc, b, bDesc,out, oDesc, stream);
        case real8U:   return flattenedNew<Op<A, uchar4>>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
        case real8I:   return flattenedNew<Op<A, char4>>(a, aDesc, b, bDesc,out, oDesc, stream, 2);
        case real16U:  return flattenedNew<Op<A, short2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case real16I:  return flattenedNew<Op<A, short2>>(a, aDesc, b, bDesc,out, oDesc, stream, 1);
        case boolean:  return flattenedNew<Op<A, bool4>>(a, aDesc, b, bDesc, out, oDesc, stream, 2);
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