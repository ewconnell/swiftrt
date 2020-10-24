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
#include "reduce_api.h"
#include "tensor.cuh"
#include "math_fn.cuh"

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
using namespace cub;

//==============================================================================
// supplemental type limits
//==============================================================================

//------------------------------------------------------------------------------
// bfloat16
template <>
struct FpLimits<bfloat16>
{
    static __host__ __device__ __forceinline__ bfloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<bfloat16&>(max_word);
    }

    static __host__ __device__ __forceinline__ bfloat16 Lowest() {
        unsigned short lowest_word = 0x0080;
        return reinterpret_cast<bfloat16&>(lowest_word);
    }
};

template <> struct NumericTraits<bfloat16> : 
    BaseTraits<FLOATING_POINT, true, false, unsigned short, bfloat16> {};

//------------------------------------------------------------------------------
// Complex<float>
template <>
struct FpLimits<Complex<float>>
{
    static __host__ __device__ __forceinline__ Complex<float> Max() {
        return Complex<float>(FLT_MAX, FLT_MAX);
    }

    static __host__ __device__ __forceinline__ Complex<float> Lowest() {
        return Complex<float>(FLT_MIN, FLT_MIN);
    }
};

template <> struct NumericTraits<Complex<float>> : 
    BaseTraits<FLOATING_POINT, true, false, uint64_t, Complex<float>> {};

//------------------------------------------------------------------------------
// Complex<float16>
template <>
struct FpLimits<Complex<float16>>
{
    static __host__ __device__ __forceinline__ Complex<float16> Max() {
        return Complex<float16>(FLT_MAX, FLT_MAX);
    }

    static __host__ __device__ __forceinline__ Complex<float16> Lowest() {
        return Complex<float16>(FLT_MIN, FLT_MIN);
    }
};

template <> struct NumericTraits<Complex<float16>> : 
    BaseTraits<FLOATING_POINT, true, false, uint32_t, Complex<float16>> {};


//==============================================================================
// supplemental function delegating macros
//==============================================================================

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct AbsSumOp {
    struct AbsSum {
        template <typename T>
        __device__ __forceinline__
        T operator()(const T &a, const T &b) const {
            return abs(a) + abs(b);
        }
    };

    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        typedef typename std::remove_const_t<std::remove_pointer_t<InputIteratorT>> T;
        AbsSum abs_sum;
        return cub::DeviceReduce::
        Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, count, abs_sum, T(), stream);
    }
};

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct AnyOp {
    struct LogicalOr {
        template <typename T>
        __device__ __forceinline__
        T operator()(const T &a, const T &b) const {
            return b || a;
        }
    };

    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        LogicalOr logical_or;
        return cub::DeviceReduce::
        Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, count, logical_or, false, stream);
    }
};

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct AllOp {
    struct LogicalAnd {
        template <typename T>
        __device__ __forceinline__
        T operator()(const T &a, const T &b) const {
            return b && a;
        }
    };

    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        LogicalAnd logical_and;
        return cub::DeviceReduce::
        Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, count, logical_and, true, stream);
    }
};

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct MinOp {
    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        return cub::DeviceReduce::
        Min(d_temp_storage, temp_storage_bytes, d_in, d_out, count, stream);
    }
};

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct MaxOp {
    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        return cub::DeviceReduce::
        Max(d_temp_storage, temp_storage_bytes, d_in, d_out, count, stream);
    }
};

//------------------------------------------------------------------------------
template<typename InputIteratorT, typename OutputIteratorT>
struct SumOp {
    __forceinline__ static cudaError_t op(
        void*&          d_temp_storage,
        size_t&         temp_storage_bytes,
        InputIteratorT  d_in,
        OutputIteratorT d_out,
        int             count,
        cudaStream_t    stream
    ) {
        return cub::DeviceReduce::
        Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, count, stream);
    }
};

//==============================================================================

template<template<typename A, typename Out> class Operator, typename T>
cudaError_t reduce(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cub::CachingDeviceAllocator& allocator,
    cudaStream_t stream
) {
    const T* a = static_cast<const T*>(pA);
    T* out = static_cast<T*>(pOut);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    int count = aDesc.count;
    typedef Operator<const T*, T*> Op;

    CubDebugExit(Op::op(d_temp_storage, temp_storage_bytes, a, out, count, stream));
    CubDebugExit(allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, stream));
    CubDebugExit(Op::op(d_temp_storage, temp_storage_bytes, a, out, count, stream));
    CubDebugExit(allocator.DeviceFree(d_temp_storage));
    return cudaSuccess;
}

template<template<typename I, typename O> class Op>
cudaError_t selectType(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cub::CachingDeviceAllocator& allocator,
    cudaStream_t stream
) {
    if (!(aDesc.isDense() && oDesc.isDense())) return cudaErrorNotSupported;
    switch(aDesc.type) {
        case real32F:  return reduce<Op, float>(a, aDesc, out, oDesc, allocator, stream);
        case real16F:  return reduce<Op, float16>(a, aDesc, out, oDesc, allocator, stream);
        case real16BF: return reduce<Op, bfloat16>(a, aDesc, out, oDesc, allocator, stream);
        case real64F:  return reduce<Op, double>(a, aDesc, out, oDesc, allocator, stream);
        case real32I:  return reduce<Op, int32_t>(a, aDesc, out, oDesc, allocator, stream);
        case real8U:   return reduce<Op, uint8_t>(a, aDesc, out, oDesc, allocator, stream);
        case real8I:   return reduce<Op, int8_t>(a, aDesc, out, oDesc, allocator, stream);
        case real16U:  return reduce<Op, uint16_t>(a, aDesc, out, oDesc, allocator, stream);
        case real16I:  return reduce<Op, int16_t>(a, aDesc, out, oDesc, allocator, stream);
        case boolean:  return reduce<Op, bool>(a, aDesc, out, oDesc, allocator, stream);
        case complex16F: return reduce<Op, Complex<half>>(a, aDesc, out, oDesc, allocator, stream);
        case complex32F: return reduce<Op, Complex<float>>(a, aDesc, out, oDesc, allocator, stream);
        default: return cudaErrorNotSupported;
    }
}

