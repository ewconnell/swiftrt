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
    __DEVICE_INLINE__ void operator()(const A& a, Out& out) const { \
        if constexpr (conforms()) out = name(a); \
    } \
    typedef typename packed<A>::type PA; \
    typedef typename matching_packed<PA,Out>::type POut; \
    typedef OpName<PA,POut> packed; \
};

//==============================================================================
// kernels
//==============================================================================

template<typename Op, typename IterA, typename IterOut>
__global__ void map(const Op op, const IterA iterA, IterOut iterOut) {
    auto p = IterOut::Logical(blockIdx, blockDim, threadIdx);
    if (iterOut.isInBounds(p)) op(iterA[p], iterOut[p]);
}

//==============================================================================
/// flattened
template<typename Op>
static inline cudaError_t flattened(
    const void* pA,
    void* pOut,
    uint32_t count,
    cudaStream_t stream
) {
    if constexpr (Op::conforms()) {
        CudaKernelPreCheck(stream);
        using A = const typename Op::A;
        using Out = typename Op::Out;
        A* a = static_cast<A*>(pA);
        Out* out = static_cast<Out*>(pOut);

        auto iterA = Flat(a, count);
        auto iterO = Flat(out, count);

        // get tile and grid size for launch
        dim3 tile = tileSize(iterO.count);
        dim3 grid = gridSize(iterO.count, tile);

        map<<<grid, tile, 0, stream>>>(Op(), iterA, iterO);
        return CudaKernelPostCheck(stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// initIndex tensorA
template<typename Op, int Rank,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterO>
static inline cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    using A = const typename Op::A;
    using Out = typename Op::Out;
    A* a = static_cast<A*>(pA);
    Out* out = static_cast<Out*>(pOut);

    auto iterA = IterA<A*, Rank>(a, aDesc);
    auto iterO = IterO<Out*, Rank>(out, oDesc);

    // get tile and grid size for launch
    dim3 tile = tileSize<Rank>(iterO.shape);
    dim3 grid = gridSize<Rank>(iterO.shape, tile);

    map<<<grid, tile, 0, stream>>>(Op(), iterA, iterO);
    return cudaSuccess;
}

//==============================================================================
// selectRank
template<typename Op,
    template<typename P, int R> class IterA,
    template<typename P, int R> class IterO>
static inline cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.rank == oDesc.rank);
    switch(oDesc.rank) {
    case 1: return initIndex<Op,1,IterA,IterO>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,2,IterA,IterO>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,3,IterA,IterO>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIter
template<typename Op>
static inline cudaError_t selectIter(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // the types are now known, so only generate code
    // when operator/type conformance is valid
    if constexpr (Op::conforms()) {
        return selectRank<Op,Strided,Strided>(a, aDesc, out, oDesc, stream);
    }
    return cudaErrorNotSupported;
}

//==============================================================================
// selectOut

// flat
template<template<typename A, typename O> class Op, typename A>
static inline cudaError_t selectOut(
    const void* a,
    srtDataType otype,
    void* out,
    size_t count, 
    cudaStream_t stream
) {
    switch(otype) {
    case real32F:  return flattened<Op<A,float>>(a, out, count, stream);
    case real64F:  return flattened<Op<A,double>>(a, out, count, stream);
    case real32I:  return flattened<Op<A,int32_t>>(a, out, count, stream);

    case real16F:
        if constexpr (canPack<A,float16>()) {
            return flattened<typename Op<A,float16>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,float16>>(a, out, count, stream);
        }
    case real16BF:
        if constexpr (canPack<A,bfloat16>()) {
            return flattened<typename Op<A,bfloat16>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,bfloat16>>(a, out, count, stream);
        }
    case real8U:
        if constexpr (canPack<A,uint8_t>()) {
            return flattened<typename Op<A,uint8_t>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,uint8_t>>(a, out, count, stream);
        }
    case real8I:
        if constexpr (canPack<A,int8_t>()) {
            return flattened<typename Op<A,int8_t>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,int8_t>>(a, out, count, stream);
        }
    case real16U:
        if constexpr (canPack<A,uint16_t>()) {
            return flattened<typename Op<A,uint16_t>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,uint16_t>>(a, out, count, stream);
        }
    case real16I:
        if constexpr (canPack<A,int16_t>()) {
            return flattened<typename Op<A,int16_t>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,int16_t>>(a, out, count, stream);
        }
    case boolean:
        if constexpr (canPack<A,bool>()) {
            return flattened<typename Op<A,bool>::packed>(a, out, count, stream);
        } else {
            return flattened<Op<A,bool>>(a, out, count, stream);
        }

    case complex32F:  return flattened<Op<A,Complex<float>>>(a, out, count, stream);
    case complex16F:  return flattened<Op<A,Complex<float16>>>(a, out, count, stream);
    case complex16BF: return flattened<Op<A,Complex<bfloat16>>>(a, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}

// strided
template<template<typename T, typename O> class Op, typename A>
static inline cudaError_t selectOut(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectIter<Op<A,float>>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<A,double>>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<A,int32_t>>(a, aDesc, out, oDesc, stream);

    case real16F:  return selectIter<Op<A,half>>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<A,bfloat16>>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<A,uint8_t>>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<A,int8_t>>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<A,uint16_t>>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<A,int16_t>>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<A,bool>>(a, aDesc, out, oDesc, stream);

    case complex32F:  return selectIter<Op<A,Complex<float>>>(a, aDesc, out, oDesc, stream);
    case complex16F:  return selectIter<Op<A,Complex<float16>>>(a, aDesc, out, oDesc, stream);
    case complex16BF: return selectIter<Op<A,Complex<bfloat16>>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select flat
// converts from dynamic to static type and delegates for stride selection

// input and output are the same type
template<template<typename A, typename O> class Op>
static inline cudaError_t select(
    srtDataType atype,
    const void* a,
    void* out,
    size_t count, 
    cudaStream_t stream
) {
    switch(atype) {
    case real32F:  return flattened<Op<float,float>>(a, out, count, stream);
    case real64F:  return flattened<Op<double,double>>(a, out, count, stream);
    case real32I:  return flattened<Op<int32_t,int32_t>>(a, out, count, stream);

    // recast types that are smaller than 32 bit to their packed simd form
    case real16F:  return flattened<typename Op<float16,float16>::packed>(a, out, count, stream);
    case real16BF: return flattened<typename Op<bfloat16,bfloat16>::packed>(a, out, count, stream);
    case real8U:   return flattened<typename Op<uint8_t,uint8_t>::packed>(a, out, count, stream);
    case real8I:   return flattened<typename Op<int8_t,int8_t>::packed>(a, out, count, stream);
    case real16U:  return flattened<typename Op<uint16_t,uint16_t>::packed>(a, out, count, stream);
    case real16I:  return flattened<typename Op<int16_t,int16_t>::packed>(a, out, count, stream);
    case boolean:  return flattened<typename Op<bool,bool>::packed>(a, out, count, stream);

    case complex32F:  return flattened<Op<Complex<float>, Complex<float>>>(a, out, count, stream);
    case complex16F:  return flattened<Op<Complex<float16>, Complex<float16>>>(a, out, count, stream);
    case complex16BF: return flattened<Op<Complex<bfloat16>, Complex<bfloat16>>>(a, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}

// input and output can be different type
// like for casting or Complex Abs
template<template<typename A, typename O> class Op>
static inline cudaError_t select(
    srtDataType atype,
    const void* a,
    srtDataType otype,
    void* out,
    size_t count, 
    cudaStream_t stream
) {
    switch(atype) {
    case real32F:  return selectOut<Op,float>(a, otype, out, count, stream);
    case real64F:  return selectOut<Op,double>(a, otype, out, count, stream);
    case real32I:  return selectOut<Op,int32_t>(a, otype, out, count, stream);

    case real16F:  return selectOut<Op,float16>(a, otype, out, count, stream);
    case real16BF: return selectOut<Op,bfloat16>(a, otype, out, count, stream);
    case real8U:   return selectOut<Op,uint8_t>(a, otype, out, count, stream);
    case real8I:   return selectOut<Op,int8_t>(a, otype, out, count, stream);
    case real16U:  return selectOut<Op,uint16_t>(a, otype, out, count, stream);
    case real16I:  return selectOut<Op,int16_t>(a, otype, out, count, stream);
    case boolean:  return selectOut<Op,bool>(a, otype, out, count, stream);

    // case complex32F:  return selectOut<Op, Complex<float>>(a, otype, out, count, stream);
    // case complex16F:  return selectOut<Op, Complex<float16>>(a, otype, out, count, stream);
    // case complex16BF: return selectOut<Op, Complex<bfloat16>>(a, otype, out, count, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// select strided
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
    case real32F:  return selectIter<Op<float,float>>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectIter<Op<half,half>>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectIter<Op<bfloat16,bfloat16>>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectIter<Op<double,double>>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectIter<Op<int32_t,int32_t>>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectIter<Op<uint8_t,uint8_t>>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectIter<Op<int8_t,int8_t>>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectIter<Op<uint16_t,uint16_t>>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectIter<Op<int16_t,int16_t>>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectIter<Op<bool,bool>>(a, aDesc, out, oDesc, stream);

    case complex16F:  return selectIter<Op<Complex<float16>,Complex<float16>>>(a, aDesc, out, oDesc, stream);
    case complex32F:  return selectIter<Op<Complex<float>,Complex<float>>>(a, aDesc, out, oDesc, stream);
    case complex16BF: return selectIter<Op<Complex<bfloat16>,Complex<bfloat16>>>(a, aDesc, out, oDesc, stream);
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
    case real16F:  return selectOut<Op, half>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectOut<Op, bfloat16>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectOut<Op, double>(a, aDesc, out, oDesc, stream);
    case real32I:  return selectOut<Op, int32_t>(a, aDesc, out, oDesc, stream);
    case real8U:   return selectOut<Op, uint8_t>(a, aDesc, out, oDesc, stream);
    case real8I:   return selectOut<Op, int8_t>(a, aDesc, out, oDesc, stream);
    case real16U:  return selectOut<Op, uint16_t>(a, aDesc, out, oDesc, stream);
    case real16I:  return selectOut<Op, int16_t>(a, aDesc, out, oDesc, stream);
    case boolean:  return selectOut<Op, bool>(a, aDesc, out, oDesc, stream);

    case complex32F:  return selectOut<Op, Complex<float>>(a, aDesc, out, oDesc, stream);
    case complex16F:  return selectOut<Op, Complex<float16>>(a, aDesc, out, oDesc, stream);
    case complex16BF: return selectOut<Op, Complex<bfloat16>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
