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
#include "kernelHelpers.h"
#include "index.h"

//==============================================================================
// math function additions for consistency
//==============================================================================

//------------------------------------------------------------------------------
// Float16
__device__ inline static __half abs(const __half& x) {
    return __half(abs(float(x)));
}

__device__ inline static __half2 abs(const __half2& x) { return __habs2(x); }

//------------------------------------------------------------------------------
// BFloat16
// __device__ inline static __nv_bfloat162 abs(const __nv_bfloat162& x) {
//     return __float2bfloat16_rn(abs(__bfloat162float(x)));
// }

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
__device__ inline static __nv_bfloat162 abs(const __nv_bfloat162& x) {
    return __habs2(x);
}
#else
__device__ inline __nv_bfloat162 abs(const __nv_bfloat162& x) {
    __nv_bfloat162 c;
    c.x = __float2bfloat16_rn(abs(__bfloat162float(x.x)));
    c.y = __float2bfloat16_rn(abs(__bfloat162float(x.y)));
    return c;
}
#endif

//==============================================================================
// ops
//==============================================================================

template<typename E>
struct Abs {
    __device__ inline static E op(const E& x) { return abs(x); }
};

//==============================================================================
// kernels
//==============================================================================

//--------------------------------------
/// flat
template<typename F, typename E, int R,
         template<int U> class IndexX,
         template<int U> class IndexO>
__global__ void mapFlat(
    const E *x, const IndexX<R> indexX,
    E *out, const IndexO<R> indexO 
) {
    auto position = Logical<R>(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ix = indexX.linear(position);
        int io = indexO.linear(position);
        out[io] = F::op(x[ix]);
    }
}

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// map
template<template<typename U> class Op, typename E>
static cudaError_t map(
    const void* pX, const TensorDescriptor& xDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0 
) {
    typedef Op<E> F;
    E* out = static_cast<E*>(pOut);
    const E* x = static_cast<const E*>(pX);

    // get tile and grid size for launch
    dim3 tile = tileSize<1>(oDesc);
    dim3 grid = gridSize<1>(oDesc, tile);

    if (xDesc.isDense()) {
        mapFlat<F,E,1,Flat,Flat><<<grid, tile, 0, stream>>>(
            x, Flat<1>(xDesc), out, Flat<1>(oDesc));

    } else {
        // TODO
        return cudaErrorNotSupported;
    }
    return cudaSuccess;
}

//------------------------------------------------------------------------------
// selectType
// converts from dynamic to static type and delegates for stride selection
template<template<typename U> class Op>
static cudaError_t selectType(
    const void* x, const srtTensorDescriptor* pxDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // statically cast types from C interface to use with c++ templates
    const TensorDescriptor& xDesc = static_cast<const TensorDescriptor&>(*pxDesc);
    const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*poDesc);

    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(xDesc.order == oDesc.order && oDesc.isDense());
    // must be same data type and rank, and output is dense
    assert(xDesc.type == oDesc.type && xDesc.rank == oDesc.rank);

    switch(oDesc.type) {
        case CUDA_R_32F: return map<Op, float>(x, xDesc, out, oDesc, stream);
        case CUDA_R_16BF:
            if (xDesc.isDense()) {
                return map<Op, __nv_bfloat162>(x, xDesc, out, oDesc, stream, 1);
            } else {
                return map<Op, __nv_bfloat16>(x, xDesc, out, oDesc, stream);
            }
        case CUDA_R_16F:
            if (xDesc.isDense()) {
                return map<Op, __half2>(x, xDesc, out, oDesc, stream, 1);
            } else {
                return map<Op, __half>(x, xDesc, out, oDesc, stream);
            }
        case CUDA_R_8I:  return map<Op, int8_t>(x, xDesc, out, oDesc, stream);
        case CUDA_R_8U:  return map<Op, uint8_t>(x, xDesc, out, oDesc, stream);
        case CUDA_R_16I: return map<Op, int16_t>(x, xDesc, out, oDesc, stream);
        case CUDA_R_16U: return map<Op, uint16_t>(x, xDesc, out, oDesc, stream);
        case CUDA_R_64F: return map<Op, double>(x, xDesc, out, oDesc, stream);
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
    return cudaErrorNotSupported;
}

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

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
    return cudaErrorNotSupported;
}

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

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

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

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
    return cudaErrorNotSupported;
}

cudaError_t srtPowN(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtRoot(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
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
