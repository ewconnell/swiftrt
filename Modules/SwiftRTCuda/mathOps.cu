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

template<typename T> struct OpBase { typedef T Element; };

#define MATHOP(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
    __device__ inline static T op(const T& a) { return name(a); } \
}; \

#define MATHOP2N(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
    __device__ inline static T op(const T& a, const int n) { return name(a, n); } \
}; \

#define MATHOP2(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
    __device__ inline static T op(const T& a, const T& b) { return name(a, b); } \
}; \


MATHOP(Abs, abs)
MATHOP(Acos, acos)
MATHOP(Acosh, acosh)
MATHOP(Asin, asin)
MATHOP(Asinh, asinh)
MATHOP(Atan, atan)
MATHOP2(Atan2, atan2)
MATHOP(Atanh, atanh)
MATHOP(Cos, cos)
MATHOP(Cosh, cosh)
MATHOP(Erf, erf)
MATHOP(Erfc, erfc)
MATHOP(Exp, exp)
MATHOP(Exp2, exp2)
MATHOP(Exp10, exp10)
MATHOP(ExpMinusOne, expm1)
MATHOP(Gamma, tgamma)
MATHOP2(Hypot, hypot)
MATHOP(Log, log)
MATHOP(LogOnePlus, log1p)
MATHOP(Log2, log2)
MATHOP(Log10, log10)
MATHOP(LogGamma, lgamma)
MATHOP(Neg, neg)
MATHOP2(Pow, pow)
MATHOP2N(PowN, pow)
MATHOP2N(Root, root)
MATHOP(Sigmoid, sigmoid)



//==============================================================================
// kernels
//==============================================================================

// one parameter ops
template<typename Op, typename Element, typename IndexA, typename IndexO>
__global__ void mapA(
    const Element *a, const IndexA indexA,
    Element *out, const IndexO indexO 
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia]);
    }
}

// two parameter ops
template<typename Op, typename Element, 
         typename IndexA, typename IndexB, typename IndexO>
__global__ void mapAB(
    const Element *a, const IndexA indexA,
    const Element *b, const IndexB indexB,
    Element *out, const IndexO indexO 
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia], b[ib]);
    }
}

// op(A, Scalar)
template<typename Op, typename Element, typename Scalar,
         typename IndexA, typename IndexO>
__global__ void mapAScalar(
    const Element *a, const IndexA indexA, Scalar value,
    Element *out, const IndexO indexO 
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia], value);
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
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<Op,Element,Flat,Flat>
        <<<grid, tile, 0, stream>>>(a, Flat(aDesc), out, Flat(oDesc));

    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// flattened tensorA Scalar
template<typename Op, typename Scalar>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
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

    mapAScalar<Op,Element,Scalar,Flat,Flat>
        <<<grid, tile, 0, stream>>>(a, Flat(aDesc), value, out, Flat(oDesc));

    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// flattened tensorA tensorB
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    const Element* b = static_cast<const Element*>(pB);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapAB<Op,Element,Flat,Flat,Flat><<<grid, tile, 0, stream>>>
        (a, Flat(aDesc), b, Flat(bDesc), out, Flat(oDesc));

    return cudaSuccess;
}

//==============================================================================
// initIndex tensorA
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapA<Op,Element,IndexA,IndexO>
        <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), out, IndexO(oDesc));

    return cudaSuccess;
}

// initIndex tensorA Scalar
template<typename Op, typename Scalar, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAScalar<Op,Element,Scalar,IndexA,IndexO>
        <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), value, out, IndexO(oDesc));

    return cudaSuccess;
}

// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    typedef typename Op::Element Element;
    const Element* a = static_cast<const Element*>(pA);
    const Element* b = static_cast<const Element*>(pB);
    Element* out = static_cast<Element*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAB<Op,Element,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>
        (a, IndexA(aDesc), b, IndexB(bDesc), out, IndexO(oDesc));

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
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA Scalar
template<typename Op, typename Scalar>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    Scalar value,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Scalar,Strided<1>,Strided<1>>(a, aDesc, value, out, oDesc, stream);
    case 2: return initIndex<Op,Scalar,Strided<2>,Strided<2>>(a, aDesc, value, out, oDesc, stream);
    case 3: return initIndex<Op,Scalar,Strided<3>,Strided<3>>(a, aDesc, value, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// two parameter selectRank AB
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == bDesc.order && aDesc.order == oDesc.order &&
        oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type &&
        aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);

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
// selectFloatingStrided tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case CUDA_R_32F:  return selectRank<Op<float>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16F:  return selectRank<Op<__half>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_16BF: return selectRank<Op<__nv_bfloat16>>(a, aDesc, out, oDesc, stream);
    case CUDA_R_64F:  return selectRank<Op<double>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectFloatingStrided tensorA Scalar
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op, typename Scalar>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    Scalar value,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case CUDA_R_32F:  return selectRank<Op<float>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case CUDA_R_16F:  return selectRank<Op<__half>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case CUDA_R_16BF: return selectRank<Op<__nv_bfloat16>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case CUDA_R_64F:  return selectRank<Op<double>,Scalar>(a, aDesc, value, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectFloatingStrided tensorA tensorB
template<template<typename T> class Op>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case CUDA_R_32F:  return selectRank<Op<float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case CUDA_R_16F:  return selectRank<Op<__half>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case CUDA_R_16BF: return selectRank<Op<__nv_bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case CUDA_R_64F:  return selectRank<Op<double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectAnyStrided
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectAnyStrided(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // check float types first
    auto status = selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        switch(oDesc.type) {
        case CUDA_R_32I:  return selectRank<Op<int32_t>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        return status;
    }
}

//==============================================================================
// selectFloating tensorA
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

    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return flattened<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return flattened<Op<__half>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_16BF: return flattened<Op<__nv_bfloat16>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_64F:  return flattened<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// selectFloating tensorA Scalar
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op, typename Scalar>
static cudaError_t selectFloating(
    const void* a, const srtTensorDescriptor* paDesc,
    Scalar value,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to c++ type
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, value, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return flattened<Op<float>,Scalar>(a, aDesc, value, out, oDesc, stream);
        case CUDA_R_16F:  return flattened<Op<__half>,Scalar>(a, aDesc, value, out, oDesc, stream, 1);
        case CUDA_R_16BF: return flattened<Op<__nv_bfloat16>,Scalar>(a, aDesc, value, out, oDesc, stream, 1);
        case CUDA_R_64F:  return flattened<Op<double>,Scalar>(a, aDesc, value, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// selectFloating tensorA tensorB
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloating(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to c++ type
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pbDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return flattened<Op<float>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case CUDA_R_16F:  return flattened<Op<__half>>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case CUDA_R_16BF: return flattened<Op<__nv_bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case CUDA_R_64F:  return flattened<Op<double>>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// selectAny
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectAny(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    auto status = selectFloating<Op>(a, paDesc, out, poDesc, stream);
    if (status == cudaErrorNotSupported) {
        // statically cast types from C interface to c++ type
        const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
        const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

        if (aDesc.isStrided()) {
            return selectAnyStrided<Op>(a, aDesc, out, oDesc, stream);
        } else {
            switch(oDesc.type) {
            case CUDA_R_32I:  return flattened<Op<int32_t>>(a, aDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
    }
}

//------------------------------------------------------------------------------
// selectFloatingPacked
// converts from dynamic to static type. This is called for operators that
// have native packed implementations such has __half2 or __nv_bfloat162
template<template<typename T> class Op>
static cudaError_t selectFloatingPacked(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to c++ type
    const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case CUDA_R_32F:  return flattened<Op<float>>(a, aDesc, out, oDesc, stream);
        case CUDA_R_16F:  return flattened<Op<__half2>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_16BF: return flattened<Op<__nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case CUDA_R_64F:  return flattened<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//------------------------------------------------------------------------------
// selectAnyPacked
// converts from dynamic to static type. This is called for operators that
// have native packed implementations such has __half2 or __nv_bfloat162
template<template<typename T> class Op>
static cudaError_t selectAnyPacked(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    auto status = selectFloatingPacked<Op>(a, paDesc, out, poDesc, stream);

    if (status == cudaErrorNotSupported) {
        // statically cast types from C interface to c++ type
        const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*paDesc);
        const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

        if (aDesc.isStrided()) {
            return selectAnyStrided<Op>(a, aDesc, out, oDesc, stream);
        } else {
            switch(oDesc.type) {
            case CUDA_R_32I:  return flattened<Op<int32_t>>(a, aDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
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
    return selectAnyPacked<Abs>(x, xDesc, out, oDesc, stream);
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

cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Asin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Asinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Atan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan2(
    const void* y, const srtTensorDescriptor* yDesc,
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // y comes first
    return selectFloating<Atan2>(y, yDesc, x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Atanh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Cos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Cosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Erf>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Erfc>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<ExpMinusOne>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Gamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtHypot(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Hypot>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtLog(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<LogOnePlus>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<LogGamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectAny<Neg>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtPow(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Pow>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtPowN(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<PowN>(x, xDesc, int(n), out, oDesc, stream);
}

cudaError_t srtRoot(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Root>(x, xDesc, int(n), out, oDesc, stream);
}

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Sigmoid>(x, xDesc, out, oDesc, stream);
}

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
