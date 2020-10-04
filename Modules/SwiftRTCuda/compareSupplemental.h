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
#include "mathSupplemental.h"

//==============================================================================
// supplemental logical functions
//==============================================================================

//------------------------------------------------------------------------------
// andElements 
__device__ inline bool andElements(const bool& a, const bool& b) {
    return a & b;
}

__device__ inline bool4 andElements(const bool4& a, const bool4& b) {
    return bool4(UINT_CREF(a) & UINT_CREF(b));
}

//------------------------------------------------------------------------------
// orElements 
__device__ inline bool orElements(const bool& a, const bool& b) {
    return a | b;
}

__device__ inline bool4 orElements(const bool4& a, const bool4& b) {
    return bool4(UINT_CREF(a) | UINT_CREF(b));
}

//------------------------------------------------------------------------------
// almostEqual
// normally we would compare the absolute value of the difference,
// however the abs of a complex type is it's RealType and not T.
// There could be an entire special case tree where tolerance is specified
// as the RealType only, but it's unclear if it's merited.
template<typename T>
__device__ inline bool almostEqual(const T& a, const T& b, const T& tolerance) {
    T diff = a < b ? (b - a) : (a - b);
    return diff <= tolerance;
}

__device__ inline bool2 almostEqual(
    const float162& a, 
    const float162& b,
    const float162& tolerance
) {
    return bool2(almostEqual(a.x, b.x, tolerance.x), 
                 almostEqual(a.y, b.y, tolerance.y));    
}

__device__ inline bool2 almostEqual(
    const bfloat162& a, 
    const bfloat162& b,
    const bfloat162& tolerance
) {
    return bool2(almostEqual(a.x, b.x, tolerance.x), 
                 almostEqual(a.y, b.y, tolerance.y));    
}

//------------------------------------------------------------------------------
// equal
template<typename T>
__device__ inline bool equal(const T& a, const T& b) {
    return a == b;
}

__device__ inline bool2 equal(const float162& a, const float162& b) {
    return bool2(__heq2(a,b));
}

__device__ inline bool2 equal(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x == b.x, a.y == b.y);    
#else
    return bool2(__heq2(a,b));
#endif
}

// 8
__device__ inline bool4 equal(const char4& a, const char4& b) {
    return bool4(__vcmpeq4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 equal(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpeq4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 equal(const short2& a, const short2& b) {
    return bool2(__vcmpeq2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 equal(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpeq2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
// notEqual
template<typename T>
__device__ inline bool notEqual(const T& a, const T& b) {
    return a != b;
}

__device__ inline bool2 notEqual(const float162& a, const float162& b) {
    return bool2(__hne2(a, b));
}

__device__ inline bool2 notEqual(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x != b.x, a.y != b.y);    
#else
    return bool2(__hne2(a, b));
#endif
}

// 8
__device__ inline bool4 notEqual(const char4& a, const char4& b) {
    return bool4(__vcmpne4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 notEqual(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpne4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 notEqual(const short2& a, const short2& b) {
    return bool2(__vcmpne2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 notEqual(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpne2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool greater(const T& a, const T& b) {
    return a > b;
}

__device__ inline bool2 greater(const float162& a, const float162& b) {
    return bool2(__hgt2(a, b));
}

__device__ inline bool2 greater(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x > b.x, a.y > b.y);    
#else
    return bool2(__hgt2(a, b));
#endif
}

// 8
__device__ inline bool4 greater(const char4& a, const char4& b) {
    return bool4(__vcmpgts4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 greater(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpgtu4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 greater(const short2& a, const short2& b) {
    return bool2(__vcmpgts2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 greater(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpgtu2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool greaterOrEqual(const T& a, const T& b) {
    return a >= b;
}

__device__ inline bool2 greaterOrEqual(const float162& a, const float162& b) {
    return bool2(__hge2(a, b));
}

__device__ inline bool2 greaterOrEqual(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x >= b.x, a.y >= b.y);    
#else
    return bool2(__hge2(a, b));
#endif
}

// 8
__device__ inline bool4 greaterOrEqual(const char4& a, const char4& b) {
    return bool4(__vcmpges4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 greaterOrEqual(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpgeu4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 greaterOrEqual(const short2& a, const short2& b) {
    return bool2(__vcmpges2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 greaterOrEqual(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpgeu2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool less(const T& a, const T& b) {
    return a < b;
}

__device__ inline bool2 less(const float162& a, const float162& b) {
    return bool2(__hlt2(a, b));
}

__device__ inline bool2 less(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x < b.x, a.y < b.y);    
#else
    return bool2(__hlt2(a, b));
#endif
}

// 8
__device__ inline bool4 less(const char4& a, const char4& b) {
    return bool4(__vcmplts4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 less(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpltu4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 less(const short2& a, const short2& b) {
    return bool2(__vcmplts2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 less(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpltu2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool lessOrEqual(const T& a, const T& b) {
    return a <= b;
}

__device__ inline bool2 lessOrEqual(const float162& a, const float162& b) {
    return bool2(__hle2(a, b));
}

__device__ inline bool2 lessOrEqual(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    return bool2(a.x <= b.x, a.y <= b.y);    
#else
    return bool2(__hle2(a, b));
#endif
}

// 8
__device__ inline bool4 lessOrEqual(const char4& a, const char4& b) {
    return bool4(__vcmples4(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool4 lessOrEqual(const uchar4& a, const uchar4& b) {
    return bool4(__vcmpleu4(UINT_CREF(a), UINT_CREF(b)));
}

// 16
__device__ inline bool2 lessOrEqual(const short2& a, const short2& b) {
    return bool2(__vcmples2(UINT_CREF(a), UINT_CREF(b)));
}

__device__ inline bool2 lessOrEqual(const ushort2& a, const ushort2& b) {
    return bool2(__vcmpleu2(UINT_CREF(a), UINT_CREF(b)));
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline T minElements(const T& a, const T& b) {
    return a <= b ? a : b;
}

__device__ inline float162 minElements(const float162& a, const float162& b) {
#if (__CUDA_ARCH__ < 800)
    float162 v;
    v.x = a.x <= b.x ? a.x : b.x;
    v.y = a.y <= b.y ? a.y : b.y;
    return v;
#else
    return __hmin2(a, b);
#endif
}

__device__ inline bfloat162 minElements(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    bfloat162 v;
    v.x = a.x <= b.x ? a.x : b.x;
    v.y = a.y <= b.y ? a.y : b.y;
    return v;
#else
    return __hmin2(a, b);
#endif
}

__device__ inline char4 minElements(const char4& a, const char4& b) {
    unsigned m = __vmins4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, m);
}

__device__ inline uchar4 minElements(const uchar4& a, const uchar4& b) {
    unsigned m = __vminu4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, m);
}

__device__ inline short2 minElements(const short2& a, const short2& b) {
    unsigned m = __vmins2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, m);
}

__device__ inline ushort2 minElements(const ushort2& a, const ushort2& b) {
    unsigned m = __vminu2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, m);
}


//------------------------------------------------------------------------------
template<typename T>
__device__ inline T maxElements(const T& a, const T& b) {
    return a > b ? a : b;
}

__device__ inline float162 maxElements(const float162& a, const float162& b) {
#if (__CUDA_ARCH__ < 800)
    float162 v;
    v.x = a.x > b.x ? a.x : b.x;
    v.y = a.y > b.y ? a.y : b.y;
    return v;
#else
    return __hmax2(a, b);
#endif
}

__device__ inline bfloat162 maxElements(const bfloat162& a, const bfloat162& b) {
#if (__CUDA_ARCH__ < 800)
    bfloat162 v;
    v.x = a.x > b.x ? a.x : b.x;
    v.y = a.y > b.y ? a.y : b.y;
    return v;
#else
    return __hmax2(a, b);
#endif
}

__device__ inline char4 maxElements(const char4& a, const char4& b) {
    unsigned m = __vmaxs4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, m);
}

__device__ inline uchar4 maxElements(const uchar4& a, const uchar4& b) {
    unsigned m = __vmaxu4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, m);
}

__device__ inline short2 maxElements(const short2& a, const short2& b) {
    unsigned m = __vmaxs2(UINT_CREF(a), UINT_CREF(b));
    return CAST(short2, m);
}

__device__ inline ushort2 maxElements(const ushort2& a, const ushort2& b) {
    unsigned m = __vmaxu2(UINT_CREF(a), UINT_CREF(b));
    return CAST(ushort2, m);
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline T conditionalAssign(const T& a, const T& b, const bool c) {
    return c ? a : b;
}

__device__ inline 
float162 conditionalAssign(const float162& a, const float162& b, const bool2 c) {
    float162 v;
    v.x = c.x ? a.x : b.x;
    v.y = c.y ? a.y : b.y;
    return v;
}

__device__ inline 
bfloat162 conditionalAssign(const bfloat162& a, const bfloat162& b, const bool2 c) {
    bfloat162 v;
    v.x = c.x ? a.x : b.x;
    v.y = c.y ? a.y : b.y;
    return v;
}

__device__ inline 
short2 conditionalAssign(const short2& a, const short2& b, const bool2 c) {
    return make_short2(c.x ? a.x : b.x, c.y ? a.y : b.y);
}

__device__ inline 
ushort2 conditionalAssign(const ushort2& a, const ushort2& b, const bool2 c) {
    return make_ushort2(c.x ? a.x : b.x, c.y ? a.y : b.y);
}

__device__ inline 
char4 conditionalAssign(const char4& a, const char4& b, const bool4 c) {
    return make_char4(
        c.x ? a.x : b.x,
        c.y ? a.y : b.y,
        c.z ? a.z : b.z,
        c.w ? a.w : b.w);
}

__device__ inline 
uchar4 conditionalAssign(const uchar4& a, const uchar4& b, const bool4 c) {
    return make_uchar4(
        c.x ? a.x : b.x,
        c.y ? a.y : b.y,
        c.z ? a.z : b.z,
        c.w ? a.w : b.w);
}
