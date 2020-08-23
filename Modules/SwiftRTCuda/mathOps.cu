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
#include <cmath>
#include <bits/stdint-uintn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "mathOps.h"
#include "mathOpFunctions.h"
#include "kernelHelpers.h"
#include "index.h"

//==============================================================================
// ops
//==============================================================================

template<typename StoredT, typename ComputeT> struct Abs {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(abs(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Acos {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(acos(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Acosh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(acosh(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Asin {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(asin(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Asinh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(asinh(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Atan {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(atan(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Atan2 {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a, const StoredT& b) {
        return StoredT(atan2(ComputeT(a), ComputeT(b)));
    }
};

template<typename StoredT, typename ComputeT> struct Atanh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(atanh(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Cos {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(cos(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Cosh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(cosh(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Erf {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(erf(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Erfc {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(erfc(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Exp {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(exp(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Exp2 {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(exp2(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Exp10 {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(exp10(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct ExpMinusOne {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(expm1(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Gamma {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(tgamma(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Hypot {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a, const StoredT& b) {
        return StoredT(hypot(ComputeT(a), ComputeT(b)));
    }
};

template<typename StoredT, typename ComputeT> struct Log {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(log(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct LogOnePlus {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(log1p(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Log2 {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(log2(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Log10 {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(log10(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct LogGamma {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(lgamma(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Neg {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return -a;
    }
};

template<typename StoredT, typename ComputeT> struct Pow {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a, const StoredT& b) {
        return StoredT(pow(ComputeT(a), ComputeT(b)));
    }
};

template<typename StoredT, typename ComputeT> struct PowN {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a, const int& n) {
        return StoredT(scalbn(ComputeT(a), n));
    }
};

template<typename StoredT, typename ComputeT> struct Root {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a, const int& n) {
        return n == 3 ?
            StoredT(cbrt(ComputeT(a))) :
            StoredT(pow(ComputeT(a), 1/float(n)));
    }
};

template<typename StoredT, typename ComputeT> struct Sigmoid {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(1 / (1 + exp(-ComputeT(a))));
    }
};

template<typename StoredT, typename ComputeT> struct Sign {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(ComputeT(a) < ComputeT(0) ?
            ComputeT(1) : ComputeT(0));
    }
};

template<typename StoredT, typename ComputeT> struct Sin {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(sin(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Sinh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(sinh(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Sqrt {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(sqrt(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Squared {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(ComputeT(a) * ComputeT(a));
    }
};

template<typename StoredT, typename ComputeT> struct Tan {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(tan(ComputeT(a)));
    }
};

template<typename StoredT, typename ComputeT> struct Tanh {
    typedef StoredT Element;
    __device__ inline static StoredT op(const StoredT& a) {
        return StoredT(tanh(ComputeT(a)));
    }
};

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
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<Op,Element,R,IndexA,IndexO><<<grid, tile, 0, stream>>>(
        a, IndexA<R>(aDesc), 
        out, IndexO<R>(oDesc));
    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// selectIndex
template<typename Op>
static cudaError_t selectIndex(
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

    if (aDesc.isDense()) {
        return initIndex<Op,1,Flat,Flat>(a, aDesc, out, oDesc, stream);
    } else {
        switch (oDesc.rank) {
        case 1: return initIndex<Op,1,Strided,Strided>(a, aDesc, out, oDesc, stream);
        case 2: return initIndex<Op,2,Strided,Strided>(a, aDesc, out, oDesc, stream);
        case 3: return initIndex<Op,3,Strided,Strided>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }    
}

//------------------------------------------------------------------------------
// selectFloatingPoint
// converts from dynamic to static type and delegates for stride selection
template<template<typename StoredT, typename ComputeT> class Op>
static cudaError_t selectFloatingPoint(
    const void* a, const srtTensorDescriptor* aDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    switch(oDesc->type) {
    case CUDA_R_32F:  return selectIndex<Op<float,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16F:  return selectIndex<Op<__half,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16BF: return selectIndex<Op<__nv_bfloat16,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_64F:  return selectIndex<Op<double,double>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectAny
// converts from dynamic to static type and delegates for stride selection
template<template<typename Stored, typename Compute> class Op>
static cudaError_t selectAny(
    const void* a, const srtTensorDescriptor* aDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    switch(oDesc->type) {
    case CUDA_R_32F:  return selectIndex<Op<float,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16F:  return selectIndex<Op<__half,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16BF: return selectIndex<Op<__nv_bfloat16,float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_32I:  return selectIndex<Op<int32_t,int32_t>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_64F:  return selectIndex<Op<double,double>>(a, aDesc, out, oDesc, stream);
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
    return selectAny<Abs>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Acos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Acosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Asin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Asinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Atan>(x, xDesc, out, oDesc, stream);
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
    return selectFloatingPoint<Atanh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Cos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Cosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Erf>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Erfc>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Exp>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Exp2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Exp10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<ExpMinusOne>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Gamma>(x, xDesc, out, oDesc, stream);
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
    return selectFloatingPoint<Log>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<LogOnePlus>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Log2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Log10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<LogGamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Neg>(x, xDesc, out, oDesc, stream);
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

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Sigmoid>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSign(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Sign>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Sin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Sinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSqrt(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Sqrt>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSquared(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Squared>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtTan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Tan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtTanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPoint<Tanh>(x, xDesc, out, oDesc, stream);
}
