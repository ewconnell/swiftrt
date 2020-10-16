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
#include "math_c.h"
#include "float16.cuh"
#include "bfloat16.cuh"

// #include <__clang_cuda_device_functions.h>

//==============================================================================
// supplemental + - * / functions
//==============================================================================

// add
template<typename T>
__DEVICE_INLINE__ T add(const T& a, const T& b) { return a + b; }

// divide
template<typename T>
__DEVICE_INLINE__ T divide(const T& a, const T& b) { return a / b; }

// multiply
template<typename T>
__DEVICE_INLINE__ T multiply(const T& a, const T& b) { return a * b; }

// subtract
template<typename T>
__DEVICE_INLINE__ T subtract(const T& a, const T& b) { return a - b; }

//------------------------------------------------------------------------------
// multiply add
template<typename T>
__DEVICE_INLINE__ T multiplyAdd(const T& a, const T& b, const T& c) {
    return a * b + c;
}

__DEVICE_INLINE__ 
float162 multiplyAdd(const float162& a, const float162& b, const float162& c) {
#if (__CUDA_ARCH__ < 800)
    return float162(a.x * b.x + c.x, a.y * b.y + c.y);
#else
    return __hfma2(a, b, c);
#endif
}

__DEVICE_INLINE__ 
bfloat162 multiplyAdd(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
#if (__CUDA_ARCH__ < 800)
    return bfloat162(a.x * b.x + c.x, a.y * b.y + c.y);
#else
    return __hfma2(a, b, c);
#endif
}

//==============================================================================
// supplemental custom functions
//==============================================================================

// neg
template<typename T>
__DEVICE_INLINE__ T neg(const T& a) { return -a; }

//==============================================================================
// sign 
template<typename T>
__DEVICE_INLINE__ T sign(const T& a) { return a < T(0) ? T(-1) : T(1); }

__DEVICE_INLINE__ uint32_t sign(const uint32_t& a) { return 1; }
__DEVICE_INLINE__ uint16_t sign(const uint16_t& a) { return 1; }
__DEVICE_INLINE__ uint8_t sign(const uint8_t& a) { return 1; }

// uchar4
__DEVICE_INLINE__ uchar4 sign(const uchar4& a) {
    const uint32_t value = 0x01010101;
    return CAST(uchar4, value);
}

// char4
__DEVICE_INLINE__ char4 sign(const char4& a) {
    char4 out;
    out.w = a.w < 0 ? -1 : 1;
    out.x = a.x < 0 ? -1 : 1;
    out.y = a.y < 0 ? -1 : 1;
    out.z = a.z < 0 ? -1 : 1;
    return out;
}

// ushort2
__DEVICE_INLINE__ ushort2 sign(const ushort2& a) {
    const uint32_t value = 0x00010001;
    return CAST(ushort2, value);
}

__DEVICE_INLINE__ short2 sign(const short2& a) {
    short2 out;
    out.x = a.x < 0 ? -1 : 1;
    out.y = a.y < 0 ? -1 : 1;
    return out;
}

//==============================================================================
// squared 
template<typename T>
__DEVICE_INLINE__ T squared(const T& a) { return a * a; }

//------------------------------------------------------------------------------
// sigmoid Float16
template<typename T>
__DEVICE_INLINE__ T sigmoid(const T& a) { return T(1) / (T(1) + exp(-a)); }

//==============================================================================
// complex supplemental functions
//==============================================================================

// template<typename I, typename O>
// __HOSTDEVICE_INLINE__ inline O abs(const I& v) {
//     return sqrt(v.real() * v.real() + v.imaginary() * v.imaginary());
// } 

//==============================================================================
// add


//==============================================================================
// SIMD supplemental functions
//==============================================================================

//==============================================================================
// neg

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 operator-(const char4& v) {
    auto out = __vneg4(UINT_CREF(v));
    return CAST(char4, out);
}
__DEVICE_INLINE__ uchar4 operator-(const uchar4& v) { return v; }

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 operator-(const short2& v) {
    auto out = __vneg2(UINT_CREF(v));
    return CAST(short2, out);
}

//==============================================================================
// abs

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 abs(const char4& v) {
    auto out = __vabs4(UINT_CREF(v));
    return CAST(char4, out);
}
__DEVICE_INLINE__ uchar4 abs(const uchar4& v) { return v; }

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 abs(const short2& v) {
    auto out = __vabs2(UINT_CREF(v));
    return CAST(short2, out);
}
__DEVICE_INLINE__ ushort2 abs(const ushort2& v) { return v; }

//==============================================================================
// add

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 operator+(const char4& a, const char4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, out);
}

__DEVICE_INLINE__ uchar4 operator+(const uchar4& a, const uchar4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, out);
}

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 operator+(const short2& a, const short2& b) {
    auto out = __vadd2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, out);
}

__DEVICE_INLINE__ ushort2 operator+(const ushort2& a, const ushort2& b) {
    auto out = __vadd2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, out);
}

//==============================================================================
// subtract

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 operator-(const char4& a, const char4& b) {
    auto out = __vsub4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, out);
}

__DEVICE_INLINE__ uchar4 operator-(const uchar4& a, const uchar4& b) {
    auto out = __vsub4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, out);
}

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 operator-(const short2& a, const short2& b) {
    auto out = __vsub2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, out);
}

__DEVICE_INLINE__ ushort2 operator-(const ushort2& a, const ushort2& b) {
    auto out = __vsub2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, out);
}

//==============================================================================
// multiply

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 operator*(const char4& a, const char4& b) {
    char4 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    out.z = a.z * b.z;
    out.w = a.w * b.w;
    return out;
}

__DEVICE_INLINE__ uchar4 operator*(const uchar4& a, const uchar4& b) {
    uchar4 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    out.z = a.z * b.z;
    out.w = a.w * b.w;
    return out;
}

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 operator*(const short2& a, const short2& b) {
    short2 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    return out;
}

__DEVICE_INLINE__ ushort2 operator*(const ushort2& a, const ushort2& b) {
    ushort2 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    return out;
}

//==============================================================================
// divide

//--------------------------------------
// char4
__DEVICE_INLINE__ char4 operator/(const char4& a, const char4& b) {
    char4 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    out.z = a.z / b.z;
    out.w = a.w / b.w;
    return out;
}

__DEVICE_INLINE__ uchar4 operator/(const uchar4& a, const uchar4& b) {
    uchar4 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    out.z = a.z / b.z;
    out.w = a.w / b.w;
    return out;
}

//--------------------------------------
// short2
__DEVICE_INLINE__ short2 operator/(const short2& a, const short2& b) {
    short2 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    return out;
}

__DEVICE_INLINE__ ushort2 operator/(const ushort2& a, const ushort2& b) {
    ushort2 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    return out;
}
