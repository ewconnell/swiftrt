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
#include <assert.h>
#include "kernelHelpers.h"
#include "mathOps.h"
#include "mathSupplemental.h"
#include "index.h"

//==============================================================================
// ops
//==============================================================================

template<typename T> struct Abs {
    typedef T Element;
    static const bool native162 = true;
    __device__ inline static T op(const T& a) { return abs(a); }
};

template<typename T> struct Exp {
    typedef T Element;
    static const bool native162 = true;
    __device__ inline static T op(const T& a) { return exp(a); }
};

template<typename T> struct Acos {
    typedef T Element;
    static const bool native162 = false;
    __device__ inline static T op(const T& a) { return acos(a); }
};

template<typename T> struct Acosh {
    typedef T Element;
    static const bool native162 = false;
    __device__ inline static T op(const T& a) { return acosh(a); }
};

// template<typename StoredT, typename ComputeT> struct Cos {
//     typedef StoredT Element;
//     __device__ inline static StoredT op(const StoredT& a) {
//         return StoredT(cos(ComputeT(a)));
//     }
// };

// template<typename StoredT, typename ComputeT> struct Exp {
//     typedef StoredT Element;
//     __device__ inline static StoredT op(const StoredT& a) {
//         return StoredT(exp(ComputeT(a)));
//     }
// };

// template<typename StoredT, typename ComputeT> struct Sigmoid {
//     typedef StoredT Element;
//     __device__ inline static StoredT op(const StoredT& a) {
//         return StoredT(1 / (1 + exp(-ComputeT(a))));
//     }
// };

//==============================================================================
// kernels
//==============================================================================

// single parameter ops
template<typename Op, typename Element, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
__global__ void mapA(
    const Element *a, const IndexA<R> indexA,
    Element *out, const IndexO<R> indexO 
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia]);
    }
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// withFlatIndex
template<typename Op>
static cudaError_t withFlatIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<Op,Element,1,Flat,Flat><<<grid, tile, 0, stream>>>(
        a, Flat<1>(aDesc), out, Flat<1>(oDesc));
    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// initIndex
template<typename Op, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<R>(oDesc);
    dim3 grid = gridSize<R>(oDesc, tile);

    mapA<Op,Element,R,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc), 
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

/// withStridedIndex
template<typename Op>
static cudaError_t withStridedIndex(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,1,Strided,Strided>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,2,Strided,Strided>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,3,Strided,Strided>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//------------------------------------------------------------------------------
// selectFloating
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloating(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to c++ type
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    if (aDesc.isDense()) {
        // Types that can be packed like Float16 are retyped to their
        // packed form and count / 2 for efficiency __half --> __half2
        switch(oDesc.type) {
        case CUDA_R_32F:  return withFlatIndex<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return withFlatIndex<Op<__half2>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_16BF: return withFlatIndex<Op<__nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_64F:  return withFlatIndex<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return withStridedIndex<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return withStridedIndex<Op<__half>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16BF: return withStridedIndex<Op<__nv_bfloat16>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_64F:  return withStridedIndex<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// selectAny
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectAny(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to c++ type
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    if (aDesc.isDense()) {
        // Types that can be packed like Float16 are retyped to their
        // packed form and count / 2 for efficiency __half --> __half2
        switch(oDesc.type) {
        case CUDA_R_32F:  return withFlatIndex<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return withFlatIndex<Op<__half2>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_16BF: return withFlatIndex<Op<__nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_32I:  return withFlatIndex<Op<int32_t>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_64F:  return withFlatIndex<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return withStridedIndex<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return withStridedIndex<Op<__half>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16BF: return withStridedIndex<Op<__nv_bfloat16>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_32I:  return withStridedIndex<Op<int32_t>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_64F:  return withStridedIndex<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

// All types
cudaError_t srtAbs(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectAny<Abs>(x, xDesc, out, oDesc, stream);
}

// Must be promoted types
cudaError_t srtAcos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Acos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Acosh>(x, xDesc, out, oDesc, stream);
}

// specialized H2
cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Exp>(x, xDesc, out, oDesc, stream);
}

// custom
cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

//==============================================================================
cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtAtan2(
    const void* y, const srtTensorDescriptor* yDesc,
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // y comes first
    // return selectTypeAB<Atan2>(y, yDesc, x, xDesc, out, oDesc, stream);
    return cudaErrorNotSupported;
}

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

// cudaError_t srtCos(
//     const void* x, const srtTensorDescriptor* xDesc,
//     void* out, const srtTensorDescriptor* oDesc,
//     cudaStream_t stream)
// {
//     return cudaErrorNotSupported;
// }

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

// cudaError_t srtExp(
//     const void* x, const srtTensorDescriptor* xDesc,
//     void* out, const srtTensorDescriptor* oDesc,
//     cudaStream_t stream)
// {
//     return cudaErrorNotSupported;
// }

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtHypot(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // return selectTypeAB<Hypot>(x, xDesc, y, yDesc, out, oDesc, stream);
    return cudaErrorNotSupported;
}

cudaError_t srtLog(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtPow(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // return selectTypeAB<Pow>(x, xDesc, y, yDesc, out, oDesc, stream);
    return cudaErrorNotSupported;
}

cudaError_t srtPowN(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // return selectTypeAN<PowN>(x, xDesc, n, out, oDesc, stream);
    return cudaErrorNotSupported;
}

cudaError_t srtRoot(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // return selectTypeAN<Root>(x, xDesc, n, out, oDesc, stream);
    return cudaErrorNotSupported;
}

// cudaError_t srtSigmoid(
//     const void* x, const srtTensorDescriptor* xDesc,
//     void* out, const srtTensorDescriptor* oDesc,
//     cudaStream_t stream)
// {
//     return cudaErrorNotSupported;
// }

cudaError_t srtSign(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtSin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtSinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtSqrt(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtSquared(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtTan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtTanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}
