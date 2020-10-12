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
#include "index.h"
#include "reduce_fn.h"
#include "op1.h"
#include <cstddef>


// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtAbsmax(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtAbssum(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtAll(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtAny(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

//==============================================================================

template<typename InputIteratorT, typename OutputIteratorT>
struct sumOp {
    inline cudaError_t operator() (
        void           *d_temp_storage,
        size_t          temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        return DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                 d_in, d_out, count, stream);
    }
};

template<template<typename I, typename O> class Op, typename T>
inline cudaError_t reduce(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    const T* a = static_cast<const T*>(pA);
    T* out = static_cast<T*>(pOut);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int count = aDesc.count;
    Op<const T*, T*> op = Op<const T*, T*>();

    CubDebugExit(op(d_temp_storage, temp_storage_bytes, a, out, count, stream));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, stream));
    CubDebugExit(op(d_temp_storage, temp_storage_bytes, a, out, count, stream));
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    return cudaSuccess;
}

template<template<typename I, typename O> class Op>
cudaError_t selectType(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (!(aDesc.isDense() && oDesc.isDense())) return cudaErrorNotSupported;
    switch(aDesc.type) {
        case real32F:  return reduce<Op, float>(a, aDesc, out, oDesc, stream);
        case real16F:  return reduce<Op, float16>(a, aDesc, out, oDesc, stream);
        case real16BF: return reduce<Op, bfloat16>(a, aDesc, out, oDesc, stream);
        case real64F:  return reduce<Op, double>(a, aDesc, out, oDesc, stream);
        case real32I:  return reduce<Op, int32_t>(a, aDesc, out, oDesc, stream);
        case real8U:   return reduce<Op, uint8_t>(a, aDesc, out, oDesc, stream);
        case real8I:   return reduce<Op, int8_t>(a, aDesc, out, oDesc, stream);
        case real16U:  return reduce<Op, uint16_t>(a, aDesc, out, oDesc, stream);
        case real16I:  return reduce<Op, int16_t>(a, aDesc, out, oDesc, stream);
        case boolean:  return reduce<Op, bool>(a, aDesc, out, oDesc, stream);
        case complex32F: return reduce<Op, complexf>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
    }
}

cudaError_t srtSum(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<sumOp>(a, aDesc, out, oDesc, stream);
}

//==============================================================================

cudaError_t srtMean(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtMinElement(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtMaxElement(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtProd(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtProdNonZeros(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
