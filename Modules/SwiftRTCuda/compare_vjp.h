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
