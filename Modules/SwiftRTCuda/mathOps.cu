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
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "mathOps.h"
#include "mathOpFunctions.h"
#include "kernelHelpers.h"
#include "index.h"

//==============================================================================
// ops
//==============================================================================

template<typename E> struct Abs {
    __device__ inline static E op(const E& a) { return abs(a); }
};

template<typename E> struct Acos {
    __device__ inline static E op(const E& a) { return acos(a); }
};

template<typename E> struct Acosh {
    __device__ inline static E op(const E& a) { return acosh(a); }
};

template<typename E> struct Asin {
    __device__ inline static E op(const E& a) { return asin(a); }
};

template<typename E> struct Asinh {
    __device__ inline static E op(const E& a) { return asinh(a); }
};

template<typename E> struct Atan {
    __device__ inline static E op(const E& a) { return atan(a); }
};

template<typename E> struct Atan2 {
    __device__ inline static E op(const E& a, const E& b) { return atan2(a, b); }
};

template<typename E> struct Atanh {
    __device__ inline static E op(const E& a) { return atanh(a); }
};

template<typename E> struct Cos {
    __device__ inline static E op(const E& a) { return cos(a); }
};

template<typename E> struct Cosh {
    __device__ inline static E op(const E& a) { return cosh(a); }
};

template<typename E> struct Erf {
    __device__ inline static E op(const E& a) { return erf(a); }
};

template<typename E> struct Erfc {
    __device__ inline static E op(const E& a) { return erfc(a); }
};

template<typename E> struct Exp {
    __device__ inline static E op(const E& a) { return exp(a); }
};

template<typename E> struct Exp2 {
    __device__ inline static E op(const E& a) { return exp2(a); }
};

template<typename E> struct Exp10 {
    __device__ inline static E op(const E& a) { return exp10(a); }
};

template<typename E> struct ExpMinusOne {
    __device__ inline static E op(const E& a) { return expm1(a); }
};

template<typename E> struct Gamma {
    __device__ inline static E op(const E& a) { return tgamma(a); }
};

template<typename E> struct Hypot {
    __device__ inline static E op(const E& a, const E& b) { return hypot(a, b); }
};

template<typename E> struct Log {
    __device__ inline static E op(const E& a) { return log(a); }
};

template<typename E> struct LogOnePlus {
    __device__ inline static E op(const E& a) { return log1p(a); }
};

template<typename E> struct Log2 {
    __device__ inline static E op(const E& a) { return log2(a); }
};

template<typename E> struct Log10 {
    __device__ inline static E op(const E& a) { return log10(a); }
};

template<typename E> struct LogGamma {
    __device__ inline static E op(const E& a) { return lgamma(a); }
};

template<typename E> struct Neg {
    __device__ inline static E op(const E& a) { return -a; }
};

template<typename E> struct Pow {
    __device__ inline static E op(const E& a, const E& b) { return pow(a, b); }
};

template<typename E, typename N> struct PowN {
    __device__ inline static E op(const E& a, N n) { return scalbn(a, n); }
};

template<typename E, typename N> struct Root {
    __device__ inline static E op(const E& a, N n) {
        return n == 3 ? cbrt(a) : pow(a, 1/float(n));
    }
};

template<typename E> struct Sigmoid {
    __device__ inline static E op(const E& a) { return 1 / (1 + exp(-a)); }
};

template<typename E> struct Sign {
    __device__ inline static E op(const E& a) { return signbit(a); }
};

template<typename E> struct Sin{
    __device__ inline static E op(const E& a) { return sin(a); }
};

template<typename E> struct Sinh {
    __device__ inline static E op(const E& a) { return sinh(a); }
};

template<typename E> struct Sqrt {
    __device__ inline static E op(const E& a) { return sqrt(a); }
};

template<typename E> struct Squared {
    __device__ inline static E op(const E& a) { return a * a; }
};

template<typename E> struct Tan {
    __device__ inline static E op(const E& a) { return tan(a); }
};

template<typename E> struct Tanh {
    __device__ inline static E op(const E& a) { return tanh(a); }
};

//==============================================================================
// kernels
//==============================================================================

// for single parameter ops
template<typename F, typename E, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
__global__ void mapA(
    const E *a, const IndexA<R> indexA,
    E *out, const IndexO<R> indexO 
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = F::op(a[ia]);
    }
}

// for single parameter with scalar parameter ops
template<typename F, typename E, typename N, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
__global__ void mapA(
    const E *a, const IndexA<R> indexA,
    N n,
    E *out, const IndexO<R> indexO 
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = F::op(a[ia], n);
    }
}

// for two parameter ops
template<typename F, typename E, int R,
         template<int U> class IndexA,
         template<int U> class IndexB,
         template<int U> class IndexO>
__global__ void mapAB(
    const E *a, const IndexA<R> indexA,
    const E *b, const IndexB<R> indexB,
    E *out, const IndexO<R> indexO 
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        out[io] = F::op(a[ia], b[ib]);
    }
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// initIndexA
template<typename F, typename E, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
static cudaError_t initIndexA(
    const E* a, const TensorDescriptor& aDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // get tile and grid size for launch
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<F,E,R,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc), 
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

/// initIndexAN
template<typename F, typename E, typename N, int R,
         template<int U> class IndexA,
         template<int U> class IndexO>
static cudaError_t initIndexAN(
    const E* a, const TensorDescriptor& aDesc,
    N n,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // get tile and grid size for launch
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<F,E,N,R,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc), n, 
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

/// initIndexAB
template<typename F, typename E, int R,
         template<int U> class IndexA,
         template<int U> class IndexB,
         template<int U> class IndexO>
static cudaError_t initIndexAB(
    const E* a, const TensorDescriptor& aDesc,
    const E* b, const TensorDescriptor& bDesc,
    E* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // get tile and grid size for launch
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapAB<F,E,R,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc),
        b, IndexB<R>(bDesc),
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// selectIndexA
template<template<typename U> class Op, typename E>
static cudaError_t selectIndexA(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef Op<E> F;
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);

    if (aDesc.isDense()) {
        return initIndexA<F,E,1,Flat,Flat>(a, aDesc, out, oDesc, stream);
    } else {
        switch (oDesc.rank) {
        case 1: return initIndexA<F,E,1,Strided,Strided>(a, aDesc, out, oDesc, stream);
        case 2: return initIndexA<F,E,2,Strided,Strided>(a, aDesc, out, oDesc, stream);
        case 3: return initIndexA<F,E,3,Strided,Strided>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }    
}

/// selectIndexAN
template<template<typename U, typename V> class Op, typename E, typename N>
static cudaError_t selectIndexAN(
    const void* pA, const TensorDescriptor& aDesc,
    N n,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef Op<E, N> F;
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);

    if (aDesc.isDense()) {
        return initIndexAN<F,E,N,1,Flat,Flat>(a, aDesc, n, out, oDesc, stream);
    } else {
        switch (oDesc.rank) {
        case 1: return initIndexAN<F,E,N,1,Strided,Strided>(a, aDesc, n, out, oDesc, stream);
        case 2: return initIndexAN<F,E,N,2,Strided,Strided>(a, aDesc, n, out, oDesc, stream);
        case 3: return initIndexAN<F,E,N,3,Strided,Strided>(a, aDesc, n, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }    
}

/// selectIndexAB
template<template<typename U> class Op, typename E>
static cudaError_t selectIndexAB(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef Op<E> F;
    E* out = static_cast<E*>(pOut);
    const E* a = static_cast<const E*>(pA);
    const E* b = static_cast<const E*>(pB);

    if (aDesc.isDense()) {
        return initIndexAB<F,E,1,Flat,Flat,Flat>(a, aDesc, b, bDesc, out, oDesc, stream);
    } else {
        switch (oDesc.rank) {
        case 1: return initIndexAB<F,E,1,Strided,Strided,Strided>(a, aDesc, b, bDesc, out, oDesc, stream);
        case 2: return initIndexAB<F,E,2,Strided,Strided,Strided>(a, aDesc, b, bDesc, out, oDesc, stream);
        case 3: return initIndexAB<F,E,3,Strided,Strided,Strided>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }    
}

//------------------------------------------------------------------------------
// selectType
// converts from dynamic to static type and delegates for stride selection
template<template<typename U> class Op>
static cudaError_t selectType(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch(oDesc.type) {
        case CUDA_R_32F:  return selectIndexA<Op, float>(a, aDesc, out, oDesc, stream);
        // case CUDA_R_16BF: return selectIndex<Op, __nv_bfloat16>(a, aDesc, out, oDesc, stream);
        // case CUDA_R_16F:  return selectIndex<Op, __half>(a, aDesc, out, oDesc, stream);
        case CUDA_R_64F:  return selectIndexA<Op, double>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
    }
}

// selectType
// converts from dynamic to static type and delegates for stride selection
template<template<typename U, typename V> class Op>
static cudaError_t selectTypeAN(
    const void* a, const srtTensorDescriptor* paDesc, int n,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch(oDesc.type) {
        case CUDA_R_32F:  return selectIndexAN<Op, float, int>(a, aDesc, n, out, oDesc, stream);
        // case CUDA_R_16BF: return selectIndex<Op, __nv_bfloat16>(a, aDesc, out, oDesc, stream);
        // case CUDA_R_16F:  return selectIndex<Op, __half>(a, aDesc, out, oDesc, stream);
        case CUDA_R_64F:  return selectIndexAN<Op, double>(a, aDesc, n, out, oDesc, stream);
        default: return cudaErrorNotSupported;
    }
}

// selectTypeAB
template<template<typename U> class Op>
static cudaError_t selectTypeAB(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pbDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == bDesc.order && aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch(oDesc.type) {
    case CUDA_R_32F:  return selectIndexAB<Op, float>(a, aDesc, b, bDesc, out, oDesc, stream);
    // case CUDA_R_16BF: return selectIndex<Op, __nv_bfloat16>(a, aDesc, out, oDesc, stream);
    // case CUDA_R_16F:  return selectIndex<Op, __half>(a, aDesc, out, oDesc, stream);
    case CUDA_R_64F:  return selectIndexAB<Op, double>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtAbs(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Abs>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Acos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Acosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Asin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Asinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Atan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan2(
    const void* y, const srtTensorDescriptor* yDesc,
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // y comes first
    return selectTypeAB<Atan2>(y, yDesc, x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Atanh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Cos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Cosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Erf>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Erfc>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Exp>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Exp2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Exp10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<ExpMinusOne>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Gamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtHypot(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectTypeAB<Hypot>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtLog(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Log>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<LogOnePlus>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Log2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Log10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<LogGamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Neg>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtPow(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectTypeAB<Pow>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtPowN(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectTypeAN<PowN>(x, xDesc, n, out, oDesc, stream);
}

cudaError_t srtRoot(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectTypeAN<Root>(x, xDesc, n, out, oDesc, stream);
}

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Sigmoid>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSign(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Sign>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Sin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Sinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSqrt(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Sqrt>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSquared(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Squared>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtTan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Tan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtTanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectType<Tanh>(x, xDesc, out, oDesc, stream);
}
