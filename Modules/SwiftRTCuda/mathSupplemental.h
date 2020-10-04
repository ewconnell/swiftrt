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
#include "dispatchHelpers.h"


//==============================================================================
// supplemental function delegating macros
//==============================================================================

#define FLOAT16_NATIVE(func, native) \
__device__ inline float16 func(const float16& a) { return native(a); }

#define FLOAT162_NATIVE(func, native) \
__device__ inline float162 func(const float162& a) { return native(a); }

#if (__CUDA_ARCH__ < 800)
#define BFLOAT16_NATIVE(func, native) \
    __device__ inline bfloat16 func(const bfloat16& a) \
    { return func(float(a)); }
#else
#define BFLOAT16_NATIVE(func, native) \
    __device__ inline bfloat16 func(const bfloat16& a) { return native(a); }
#endif

#if (__CUDA_ARCH__ < 800)
#define BFLOAT162_NATIVE(func, native) \
    __device__ inline bfloat162 func(const bfloat162& a) \
    { return bfloat162(func(float(a.x)), func(float(a.y))); }
#else
#define BFLOAT162_NATIVE(func, native) \
    __device__ inline bfloat162 func(const bfloat162& a) { return native(a); }
#endif

// two input
#define FLOAT16_NATIVE2(func, native) \
__device__ inline float16 func(const float16& a, const float16& b) \
    { return native(a, b); }

#define FLOAT162_NATIVE2(func, native) \
__device__ inline float162 func(const float162& a, const float162& b) \
    { return native(a, b); }

#if (__CUDA_ARCH__ < 800)
#define BFLOAT16_NATIVE2(func, native) \
    __device__ inline bfloat16 func(const bfloat16& a, const bfloat16& b) \
    { return func(float(a), float(b)); }
#else
#define BFLOAT16_NATIVE(func, native) \
    __device__ inline bfloat16 func(const bfloat16& a, const bfloat16& b) \
    { return native(a, b); }
#endif

#if (__CUDA_ARCH__ < 800)
#define BFLOAT162_NATIVE2(func, native) \
    __device__ inline bfloat162 func(const bfloat162& a, const bfloat162& b) \
    { return bfloat162(func(float(a.x), float(b.x)), func(float(a.y), float(b.y))); }
#else
#define BFLOAT162_NATIVE2(func, native) \
    __device__ inline bfloat162 func(const bfloat162& a, const bfloat162& b) \
    { return native(a, b); }
#endif

//------------------------------------------------------------------------------
// Promotes the type to a float, does the op, then back to the type. This
// is used for ops that do not natively support half or bfloat types
#define FLOAT16_CAST(func) \
__device__ inline float16 func(const float16& a) { return func(float(a)); }

#define FLOAT162_CAST(func) \
__device__ inline float162 func(const float162& a) \
    { return float162(func(a.x), func(a.y)); }

#define BFLOAT16_CAST(func) \
__device__ inline bfloat16 func(const bfloat16& a) { return func(float(a)); }

#define BFLOAT162_CAST(func) \
__device__ inline bfloat162 func(const bfloat162& a) \
    { return bfloat162(func(float(a.x)), func(float(a.y))); }

//------------------------------------------------------------------------------
// Promotes the type to a float, does the op, then back to the type. This
// is used for ops that do not natively support half or bfloat types
#define FLOAT16_CAST2(func) \
__device__ inline float16 func(const float16& a, const float16& b) \
    { return func(float(a), float(b)); }

#define FLOAT162_CAST2(func) \
__device__ inline float162 func(const float162& a, const float162& b) \
    { return float162(func(a.x, b.x), func(a.y, b.y)); }

#define BLOAT16_CAST2(func) \
__device__ inline bfloat16 func(const bfloat16& a, const bfloat16& b) \
    { return func(float(a), float(b)); }

#define BLOAT162_CAST2(func) \
__device__ inline bfloat162 func(const bfloat162& a, const bfloat162& b) \
    { return bfloat162(func(float(a.x), float(b.x)), func(float(a.y), float(b.y))); }

//==============================================================================
// supplemental functions
//==============================================================================

// abs
FLOAT16_NATIVE(abs, __habs)
FLOAT162_NATIVE(abs, __habs2)
BFLOAT16_NATIVE(abs, __habs)
BFLOAT162_NATIVE(abs, __habs2)

// acos
FLOAT16_CAST(acos)
FLOAT162_CAST(acos)
BFLOAT16_CAST(acos)
BFLOAT162_CAST(acos)

// acosh
FLOAT16_CAST(acosh)
FLOAT162_CAST(acosh)
BFLOAT16_CAST(acosh)
BFLOAT162_CAST(acosh)

// asin
FLOAT16_CAST(asin)
FLOAT162_CAST(asin)
BFLOAT16_CAST(asin)
BFLOAT162_CAST(asin)

// asinh
FLOAT16_CAST(asinh)
FLOAT162_CAST(asinh)
BFLOAT16_CAST(asinh)
BFLOAT162_CAST(asinh)

// atan
FLOAT16_CAST(atan)
FLOAT162_CAST(atan)
BFLOAT16_CAST(atan)
BFLOAT162_CAST(atan)

// atan2
FLOAT16_CAST2(atan2)
FLOAT162_CAST2(atan2)
BLOAT16_CAST2(atan2)
BLOAT162_CAST2(atan2)

// atanh
FLOAT16_CAST(atanh)
FLOAT162_CAST(atanh)
BFLOAT16_CAST(atanh)
BFLOAT162_CAST(atanh)

// cos
FLOAT16_NATIVE(cos, hcos)
FLOAT162_NATIVE(cos, h2cos)
BFLOAT16_NATIVE(cos, hcos)
BFLOAT162_NATIVE(cos, h2cos)

// cosh
FLOAT16_CAST(cosh)
FLOAT162_CAST(cosh)
BFLOAT16_CAST(cosh)
BFLOAT162_CAST(cosh)

// erf
FLOAT16_CAST(erf)
FLOAT162_CAST(erf)
BFLOAT16_CAST(erf)
BFLOAT162_CAST(erf)

// erfc
FLOAT16_CAST(erfc)
FLOAT162_CAST(erfc)
BFLOAT16_CAST(erfc)
BFLOAT162_CAST(erfc)

// exp
FLOAT16_NATIVE(exp, hexp)
FLOAT162_NATIVE(exp, h2exp)
BFLOAT16_NATIVE(exp, hexp)
BFLOAT162_NATIVE(exp, h2exp)

// exp2
FLOAT16_NATIVE(exp2, hexp2)
FLOAT162_NATIVE(exp2, h2exp2)
BFLOAT16_NATIVE(exp2, hexp2)
BFLOAT162_NATIVE(exp2, h2exp2)

// exp10
FLOAT16_NATIVE(exp10, hexp10)
FLOAT162_NATIVE(exp10, h2exp10)
BFLOAT16_NATIVE(exp10, hexp10)
BFLOAT162_NATIVE(exp10, h2exp10)

// expm1
FLOAT16_CAST(expm1)
FLOAT162_CAST(expm1)
BFLOAT16_CAST(expm1)
BFLOAT162_CAST(expm1)

// tgamma
FLOAT16_CAST(tgamma)
FLOAT162_CAST(tgamma)
BFLOAT16_CAST(tgamma)
BFLOAT162_CAST(tgamma)

// hypot
FLOAT16_CAST2(hypot)
FLOAT162_CAST2(hypot)
BLOAT16_CAST2(hypot)
BLOAT162_CAST2(hypot)

// log
FLOAT16_NATIVE(log, hlog)
FLOAT162_NATIVE(log, h2log)
BFLOAT16_NATIVE(log, hlog)
BFLOAT162_NATIVE(log, h2log)

// log1p
FLOAT16_CAST(log1p)
FLOAT162_CAST(log1p)
BFLOAT16_CAST(log1p)
BFLOAT162_CAST(log1p)

// log2
FLOAT16_NATIVE(log2, hlog2)
FLOAT162_NATIVE(log2, h2log2)
BFLOAT16_NATIVE(log2, hlog2)
BFLOAT162_NATIVE(log2, h2log2)

// log10
FLOAT16_NATIVE(log10, hlog10)
FLOAT162_NATIVE(log10, h2log10)
BFLOAT16_NATIVE(log10, hlog10)
BFLOAT162_NATIVE(log10, h2log10)

// lgamma
FLOAT16_CAST(lgamma)
FLOAT162_CAST(lgamma)
BFLOAT16_CAST(lgamma)
BFLOAT162_CAST(lgamma)

// pow
FLOAT16_CAST2(pow)
FLOAT162_CAST2(pow)
BLOAT16_CAST2(pow)
BLOAT162_CAST2(pow)

// sin
FLOAT16_NATIVE(sin, hsin)
FLOAT162_NATIVE(sin, h2sin)
BFLOAT16_NATIVE(sin, hsin)
BFLOAT162_NATIVE(sin, h2sin)

// sinh
FLOAT16_CAST(sinh)
FLOAT162_CAST(sinh)
BFLOAT16_CAST(sinh)
BFLOAT162_CAST(sinh)

// sqrt
FLOAT16_NATIVE(sqrt, hsqrt)
FLOAT162_NATIVE(sqrt, h2sqrt)
BFLOAT16_NATIVE(sqrt, hsqrt)
BFLOAT162_NATIVE(sqrt, h2sqrt)

// tan
FLOAT16_CAST(tan)
FLOAT162_CAST(tan)
BFLOAT16_CAST(tan)
BFLOAT162_CAST(tan)

// tanh
FLOAT16_CAST(tanh)
FLOAT162_CAST(tanh)
BFLOAT16_CAST(tanh)
BFLOAT162_CAST(tanh)

//==============================================================================
// supplemental + - * / functions
//==============================================================================

// add
template<typename T>
__device__ inline T add(const T& a, const T& b) { return a + b; }
FLOAT162_NATIVE2(add, __hadd2)
BFLOAT162_NATIVE2(add, __hadd2)

// divide
template<typename T>
__device__ inline T divide(const T& a, const T& b) { return a / b; }
BLOAT162_CAST2(divide)

// multiply
template<typename T>
__device__ inline T multiply(const T& a, const T& b) { return a * b; }
FLOAT162_NATIVE2(multiply, __hmul2)
BFLOAT162_NATIVE2(multiply, __hmul2)

// subtract
template<typename T>
__device__ inline T subtract(const T& a, const T& b) { return a - b; }
FLOAT162_NATIVE2(subtract, __hsub2)
BFLOAT162_NATIVE2(subtract, __hsub2)

//------------------------------------------------------------------------------
// multiply add
template<typename T>
__device__ inline T multiplyAdd(const T& a, const T& b, const T& c) {
    return a * b + c;
}

__device__ inline 
float162 multiplyAdd(const float162& a, const float162& b, const float162& c) {
    return __hfma2(a, b, c);
}

__device__ inline 
bfloat162 multiplyAdd(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
#if (__CUDA_ARCH__ < 800)
    bfloat162 v;
    v.x = a.x * b.x + c.x;
    v.y = a.y * b.y + c.y;
    return v;
#else
    return __hfma2(a, b, c);
#endif
}

//==============================================================================
// supplemental custom functions
//==============================================================================

// neg
template<typename T>
__device__ inline T neg(const T& a) { return -a; }

FLOAT162_NATIVE(neg, __hneg2)
BFLOAT162_NATIVE(neg, __hneg2)

//==============================================================================
// sign 
template<typename T>
__device__ inline T sign(const T& a) { return a < T(0) ? T(-1) : T(1); }

__device__ inline uint16_t sign(const uint16_t& a) { return 1; }
__device__ inline uint8_t sign(const uint8_t& a) { return 1; }

// Float16
__device__ inline float162 sign(const float162& a) {
    float162 out; 
    out.x = a.x < float16(0.0f) ? float16(-1.0f) : float16(1.0f); 
    out.y = a.y < float16(0.0f) ? float16(-1.0f) : float16(1.0f); 
    return out;
}

// BFloat16
__device__ inline bfloat16 sign(const bfloat16& a) {
    return a < bfloat16(0.0f) ? bfloat16(-1.0f) : bfloat16(1.0f);
}

__device__ inline bfloat162 sign(const bfloat162& a) {
    bfloat162 out;
    out.x = a.x < bfloat16(0.0f) ? bfloat16(-1.0f) : bfloat16(1.0f);
    out.y = a.y < bfloat16(0.0f) ? bfloat16(-1.0f) : bfloat16(1.0f);
    return out;
}

// uchar4
__device__ inline uchar4 sign(const uchar4& a) {
    const uint32_t value = 0x01010101;
    return CAST(uchar4, value);
}

// char4
__device__ inline char4 sign(const char4& a) {
    char4 out;
    out.w = a.w < 0 ? -1 : 1;
    out.x = a.x < 0 ? -1 : 1;
    out.y = a.y < 0 ? -1 : 1;
    out.z = a.z < 0 ? -1 : 1;
    return out;
}

// ushort2
__device__ inline ushort2 sign(const ushort2& a) {
    const uint32_t value = 0x00010001;
    return CAST(ushort2, value);
}

__device__ inline short2 sign(const short2& a) {
    short2 out;
    out.x = a.x < 0 ? -1 : 1;
    out.y = a.y < 0 ? -1 : 1;
    return out;
}

//==============================================================================
// squared 
template<typename T>
__device__ inline T squared(const T& a) { return a * a; }

__device__ inline bfloat162 squared(const bfloat162& a) {
    bfloat162 out;
    out.x = a.x * a.x;
    out.y = a.y * a.y;
    return out;
}

//------------------------------------------------------------------------------
// sigmoid Float16
template<typename T>
__device__ inline T sigmoid(const T& a) { return T(1) / (T(1) + exp(-a)); }

__device__ inline float162 sigmoid(const float162& a) {
    const float162 one = float162(1, 1);
    return one / (one + exp(-a));
}

BFLOAT16_CAST(sigmoid)
BFLOAT162_CAST(sigmoid)

//==============================================================================
// complex supplemental functions
//==============================================================================

// template<typename I, typename O>
// __CUDA_HOSTDEVICE__ inline O abs(const I& v) {
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
__device__ inline char4 operator-(const char4& v) {
    auto out = __vneg4(UINT_CREF(v));
    return CAST(char4, out);
}
__device__ inline uchar4 operator-(const uchar4& v) { return v; }

//--------------------------------------
// short2
__device__ inline short2 operator-(const short2& v) {
    auto out = __vneg2(UINT_CREF(v));
    return CAST(short2, out);
}

//==============================================================================
// abs

//--------------------------------------
// char4
__device__ inline char4 abs(const char4& v) {
    auto out = __vabs4(UINT_CREF(v));
    return CAST(char4, out);
}
__device__ inline uchar4 abs(const uchar4& v) { return v; }

//--------------------------------------
// short2
__device__ inline short2 abs(const short2& v) {
    auto out = __vabs2(UINT_CREF(v));
    return CAST(short2, out);
}
__device__ inline ushort2 abs(const ushort2& v) { return v; }

//==============================================================================
// add

//--------------------------------------
// char4
__device__ inline char4 operator+(const char4& a, const char4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, out);
}

__device__ inline uchar4 operator+(const uchar4& a, const uchar4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, out);
}

//--------------------------------------
// short2
__device__ inline short2 operator+(const short2& a, const short2& b) {
    auto out = __vadd2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, out);
}

__device__ inline ushort2 operator+(const ushort2& a, const ushort2& b) {
    auto out = __vadd2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, out);
}

//==============================================================================
// subtract

//--------------------------------------
// char4
__device__ inline char4 operator-(const char4& a, const char4& b) {
    auto out = __vsub4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, out);
}

__device__ inline uchar4 operator-(const uchar4& a, const uchar4& b) {
    auto out = __vsub4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, out);
}

//--------------------------------------
// short2
__device__ inline short2 operator-(const short2& a, const short2& b) {
    auto out = __vsub2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, out);
}

__device__ inline ushort2 operator-(const ushort2& a, const ushort2& b) {
    auto out = __vsub2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, out);
}

//==============================================================================
// multiply

//--------------------------------------
// char4
__device__ inline char4 operator*(const char4& a, const char4& b) {
    char4 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    out.z = a.z * b.z;
    out.w = a.w * b.w;
    return out;
}

__device__ inline uchar4 operator*(const uchar4& a, const uchar4& b) {
    uchar4 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    out.z = a.z * b.z;
    out.w = a.w * b.w;
    return out;
}

//--------------------------------------
// short2
__device__ inline short2 operator*(const short2& a, const short2& b) {
    short2 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    return out;
}

__device__ inline ushort2 operator*(const ushort2& a, const ushort2& b) {
    ushort2 out;
    out.x = a.x * b.x;
    out.y = a.y * b.y;
    return out;
}

//==============================================================================
// divide

//--------------------------------------
// char4
__device__ inline char4 operator/(const char4& a, const char4& b) {
    char4 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    out.z = a.z / b.z;
    out.w = a.w / b.w;
    return out;
}

__device__ inline uchar4 operator/(const uchar4& a, const uchar4& b) {
    uchar4 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    out.z = a.z / b.z;
    out.w = a.w / b.w;
    return out;
}

//--------------------------------------
// short2
__device__ inline short2 operator/(const short2& a, const short2& b) {
    short2 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    return out;
}

__device__ inline ushort2 operator/(const ushort2& a, const ushort2& b) {
    ushort2 out;
    out.x = a.x / b.x;
    out.y = a.y / b.y;
    return out;
}
