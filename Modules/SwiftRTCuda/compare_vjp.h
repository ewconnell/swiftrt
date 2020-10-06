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
#include "compare_fn.h"


//==============================================================================
// vjpMin
template<typename T>
__device__ inline T vjpMin(const T& a, const T& b, const T& c) {
    return a <= b ? c : 0;
}

__device__ inline float16 vjpMin(const float16& a, const float16& b, const float16& c) {
    return a <= b ? c : __float2half(0);
}

__device__ inline bfloat16 vjpMin(const bfloat16& a, const bfloat16& b, const bfloat16& c) {
    return a <= b ? c : __float2bfloat16(0);
}

__device__ inline 
float162 vjpMin(const float162& a, const float162& b, const float162& c) {
    auto m = lessOrEqual(a, b);
    return init_float162(m.x ? c.x : __float2half(0),
                         m.y ? c.y : __float2half(0));
}

__device__ inline 
bfloat162 vjpMin(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
    auto m = lessOrEqual(a, b);
    return init_bfloat162(m.x ? c.x : __float2bfloat16(0),
                          m.y ? c.y : __float2bfloat16(0));
}

__device__ inline short2 vjpMin(const short2& a, const short2& b, const short2& c) {
    auto m = lessOrEqual(a, b);
    return make_short2(m.x ? c.x : 0, m.y ? c.y : 0);
}

__device__ inline ushort2 vjpMin(const ushort2& a, const ushort2& b, const ushort2& c) {
    auto m = lessOrEqual(a, b);
    return make_ushort2(m.x ? c.x : 0, m.y ? c.y : 0);
}

__device__ inline char4 vjpMin(const char4& a, const char4& b, const char4& c) {
    auto m = lessOrEqual(a, b);
    return make_char4(m.x ? c.x : 0, m.y ? c.y : 0, m.z ? c.z : 0, m.w ? c.w : 0);
}

__device__ inline uchar4 vjpMin(const uchar4& a, const uchar4& b, const uchar4& c) {
    auto m = lessOrEqual(a, b);
    return make_uchar4(m.x ? c.x : 0, m.y ? c.y : 0, m.z ? c.z : 0, m.w ? c.w : 0);
}

//==============================================================================
// vjpMax
template<typename T>
__device__ inline T vjpMax(const T& a, const T& b, const T& c) {
    return a >= b ? c : 0;
}

__device__ inline float16 vjpMax(const float16& a, const float16& b, const float16& c) {
    return a >= b ? c : __float2half(0);
}

__device__ inline bfloat16 vjpMax(const bfloat16& a, const bfloat16& b, const bfloat16& c) {
    return a >= b ? c : __float2bfloat16(0);
}

__device__ inline 
float162 vjpMax(const float162& a, const float162& b, const float162& c) {
    auto m = greaterOrEqual(a, b);
    return init_float162(m.x ? c.x : __float2half(0),
                         m.y ? c.y : __float2half(0));
}

__device__ inline 
bfloat162 vjpMax(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
    auto m = greaterOrEqual(a, b);
    return init_bfloat162(m.x ? c.x : __float2bfloat16(0),
                          m.y ? c.y : __float2bfloat16(0));
}

__device__ inline short2 vjpMax(const short2& a, const short2& b, const short2& c) {
    auto m = greaterOrEqual(a, b);
    return make_short2(m.x ? c.x : 0, m.y ? c.y : 0);
}

__device__ inline ushort2 vjpMax(const ushort2& a, const ushort2& b, const ushort2& c) {
    auto m = greaterOrEqual(a, b);
    return make_ushort2(m.x ? c.x : 0, m.y ? c.y : 0);
}

__device__ inline char4 vjpMax(const char4& a, const char4& b, const char4& c) {
    auto m = greaterOrEqual(a, b);
    return make_char4(m.x ? c.x : 0, m.y ? c.y : 0, m.z ? c.z : 0, m.w ? c.w : 0);
}

__device__ inline uchar4 vjpMax(const uchar4& a, const uchar4& b, const uchar4& c) {
    auto m = greaterOrEqual(a, b);
    return make_uchar4(m.x ? c.x : 0, m.y ? c.y : 0, m.z ? c.z : 0, m.w ? c.w : 0);
}

//==============================================================================


//------------------------------------------------------------------------------
// tensorA tensorB tensorC Out Out
template<typename Op, typename IndexA, typename IndexB,
         typename IndexC, typename IndexO>
__global__ void mapABC(
    const typename Op::A* __restrict__ a, const IndexA indexA,
    const typename Op::B* __restrict__ b, const IndexB indexB,
    const typename Op::C* __restrict__ c, const IndexC indexC,
    typename Op::Out* __restrict__ out0, const IndexO indexO0,
    typename Op::Out* __restrict__ out1, const IndexO indexO1
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO0.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int ic = indexC.linear(position);
        int io0 = indexO0.linear(position);
        int io1 = indexO1.linear(position);
        Op::op(a[ia], b[ib], c[ic], out0[io0], out1[io1]);
    }
}

//==============================================================================
// select tensorA tensorB tensorC
template<template<typename A> class Op>
static cudaError_t select(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    const void* c, const TensorDescriptor& cDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    assert(aDesc.type == bDesc.type);
    switch(aDesc.type) {
    // case real32F:  return selectC<Op, float>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real16F:  return selectC<Op, float16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real16BF: return selectC<Op, bfloat16>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real64F:  return selectC<Op, double>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real32I:  return selectC<Op, int32_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real8U:   return selectC<Op, uint8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real8I:   return selectC<Op, int8_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real16U:  return selectC<Op, uint16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case real16I:  return selectC<Op, int16_t>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case boolean:  return selectC<Op, bool>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    // case complex32F: return selectC<Op, Complex<float>>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}
