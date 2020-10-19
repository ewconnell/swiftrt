//******************************************************************************
// Copyright 2020 Google LLC
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
#include <stdint.h>
#include "float16.cuh"
#include "bfloat16.cuh"

//==============================================================================
/// bool2
struct bool2 {
    bool x, y;
    __HOSTDEVICE_INLINE__ bool2() { x = false; y = false; }
    __HOSTDEVICE_INLINE__ bool2(bool _x, bool _y) { x = _x; y = _y; }
    
    __DEVICE_INLINE__ bool2(half2 v) { x = v.x; y = v.y; }
    __DEVICE_INLINE__ bool2(bfloat162 v) { x = v.x; y = v.y; }
    __DEVICE_INLINE__ bool2(unsigned v) {
        x = v & 0xFF;
        y = (v >> 16) & 0xFF;
    }
};

//==============================================================================
/// bool4
struct bool4 {
    bool x, y, z, w;
    __HOSTDEVICE_INLINE__ bool4() {
        x = false; y = false; z = false; w = false;
    }
    __HOSTDEVICE_INLINE__ bool4(bool _x, bool _y, bool _z, bool _w) {
        x = _x; y = _y; z = _z; w = _w;
    }
    __HOSTDEVICE_INLINE__ bool4(unsigned v) {
        *this = *reinterpret_cast<const bool4*>(&v);
    }
};

//==============================================================================
// sign
__DEVICE_INLINE__ char4 sign(const char4& a) {
    return make_char4(
        a.x < 0 ? -1 : 1,
        a.y < 0 ? -1 : 1,
        a.z < 0 ? -1 : 1,
        a.w < 0 ? -1 : 1);
}

__DEVICE_INLINE__ short2 sign(const short2& a) {
    return make_short2(a.x < 0 ? -1 : 1, a.y < 0 ? -1 : 1);
}

//==============================================================================
// neg
__DEVICE_INLINE__ char4 operator-(const char4& v) {
    auto out = __vneg4(UINT_CREF(v));
    return CAST(char4, out);
}

__DEVICE_INLINE__ short2 operator-(const short2& v) {
    auto out = __vneg2(UINT_CREF(v));
    return CAST(short2, out);
}

//==============================================================================
// abs
__DEVICE_INLINE__ char4 abs(const char4& v) {
    auto out = __vabs4(UINT_CREF(v));
    return CAST(char4, out);
}

__DEVICE_INLINE__ short2 abs(const short2& v) {
    auto out = __vabs2(UINT_CREF(v));
    return CAST(short2, out);
}

//==============================================================================
// add
__DEVICE_INLINE__ char4 operator+(const char4& a, const char4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(char4, out);
}

__DEVICE_INLINE__ uchar4 operator+(const uchar4& a, const uchar4& b) {
    auto out = __vadd4(UINT_CREF(a), UINT_CREF(b));
    return CAST(uchar4, out);
}

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
__DEVICE_INLINE__ char4 operator*(const char4& a, const char4& b) {
    return make_char4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__DEVICE_INLINE__ uchar4 operator*(const uchar4& a, const uchar4& b) {
    return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__DEVICE_INLINE__ short2 operator*(const short2& a, const short2& b) {
    return make_short2(a.x * b.x, a.y * b.y);
}

__DEVICE_INLINE__ ushort2 operator*(const ushort2& a, const ushort2& b) {
    return make_ushort2(a.x * b.x, a.y * b.y);
}

//==============================================================================
// divide
__DEVICE_INLINE__ char4 operator/(const char4& a, const char4& b) {
    return make_char4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__DEVICE_INLINE__ uchar4 operator/(const uchar4& a, const uchar4& b) {
    return make_uchar4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__DEVICE_INLINE__ short2 operator/(const short2& a, const short2& b) {
    return make_short2(a.x / b.x, a.y / b.y);
}

__DEVICE_INLINE__ ushort2 operator/(const ushort2& a, const ushort2& b) {
    return make_ushort2(a.x / b.x, a.y / b.y);
}
