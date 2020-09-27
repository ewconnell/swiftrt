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

//==============================================================================
// macros to define operators with type conformance
#define IntOpA(OpName, name) \
template<typename _A, typename O> struct OpName { \
    typedef _A A; typedef O Out; \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (std::is_same<A,O>::value && std::is_integral<A>::value) { \
            out = name(a); \
        } \
    } \
};

#define FloatOpA(OpName, name) \
template<typename _A, typename O> struct OpName { \
    typedef _A A; typedef O Out; \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (std::is_same<A,O>::value && std::is_floating_point<A>::value) { \
            out = name(a); \
        } \
    } \
};

#define IntFloatOpA(OpName, name) \
template<typename _A, typename O> struct OpName { \
    typedef _A A; typedef O Out; \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (std::is_same<A,O>::value && \
            (std::is_integral<A>::value || std::is_floating_point<A>::value) { \
            out = name(a); \
        } \
    } \
};

#define IntFloatComplexOpA(OpName, name) \
template<typename _A, typename O> struct OpName { \
    typedef _A A; typedef O Out; \
    __device__ inline static void op(const A& a, O& out) { \
        if constexpr (std::is_same<A,O>::value && (std::is_integral<A>::value || \
                      std::is_floating_point<A>::value)) { \
            out = name(a); \
        } else if constexpr (std::is_same<A,Complex<float>>::value) { \
            out = name(a); \
        } \
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
__global__ void mapAO(
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
static cudaError_t flattenedAO(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    const typename Op::A* a = static_cast<const typename Op::A*>(pA);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUpNew(oDesc.count, shiftCount);
    dim3 tile = tileSizeNew(packedCount);
    dim3 grid = gridSizeNew<1>(oDesc, tile);

    mapAO<Op,Flat,Flat><<<grid, tile, 0, stream>>>(a, Flat(aDesc), out, Flat(oDesc));

    return cudaSuccess;
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
        // return selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
        return cudaErrorNotSupported;
    } else {
        switch(oDesc.type) {
        case real32F:  return flattenedAO<Op<A, float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return flattenedAO<Op<A, __half2>>(a, aDesc, out, oDesc, stream, 1);
        case real16BF: return flattenedAO<Op<A, __nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case real64F:  return flattenedAO<Op<A, double>>(a, aDesc, out, oDesc, stream);
        case real32I:  return flattenedAO<Op<A, int32_t>>(a, aDesc, out, oDesc, stream);
        case real8U:   return flattenedAO<Op<A, uchar4>>(a, aDesc, out, oDesc, stream, 2);
        case real8I:   return flattenedAO<Op<A, char4>>(a, aDesc, out, oDesc, stream, 2);
        case real16U:  return flattenedAO<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
        case real16I:  return flattenedAO<Op<A, short2>>(a, aDesc, out, oDesc, stream, 1);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// select tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename A, typename O> class Op>
static cudaError_t selectA(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.isStrided()) {
        // return selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
        return cudaErrorNotSupported;
    } else {
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
}

#endif