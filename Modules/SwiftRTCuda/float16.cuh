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
#include <cuda_fp16.h>
#include "srt_types.cuh"

//==============================================================================
/// float16
struct float16 : public __half {
    __HOSTDEVICE_INLINE__ float16() : __half(0) {}
    __HOSTDEVICE_INLINE__ float16(__half v) : __half(v) {}
    __HOSTDEVICE_INLINE__ float16(float v) : __half(__float2half(v)) {}
    __HOSTDEVICE_INLINE__ operator float() const { return __half2float(*this); }
};

//==============================================================================
/// float162
struct float162 : public __half2 {
    __HOSTDEVICE_INLINE__ float162() : __half2(0, 0) {}
    __HOSTDEVICE_INLINE__ float162(__half2 v) : __half2(v) {}
    __HOSTDEVICE_INLINE__ float162(float2 v) : __half2(__float22half2_rn(v)) {}
    __HOSTDEVICE_INLINE__ float162(float v) : __half2(__float22half2_rn(make_float2(v, v))) {}
    __HOSTDEVICE_INLINE__ float162(float x, float y) : __half2(__float22half2_rn(make_float2(x, y))) {}
    __HOSTDEVICE_INLINE__ operator float2() const { return __half22float2(*this); }
};

//==============================================================================
// supplemental math functions
//==============================================================================
__DEVICE_INLINE__ float16 operator-(const __half a) { return __hneg(a); }

__DEVICE_INLINE__ float16 abs(const __half a) { return __habs(a); }
__DEVICE_INLINE__ float16 acos(const __half a) { return acosf(a); }
__DEVICE_INLINE__ float16 acosh(const __half a) { return acoshf(a); }
__DEVICE_INLINE__ float16 asin(const __half a) { return asinf(a); }
__DEVICE_INLINE__ float16 asinh(const __half a) { return asinhf(a); }
__DEVICE_INLINE__ float16 atan(const __half a) { return atanf(a); }
__DEVICE_INLINE__ float16 atan2(const __half a, const __half b) { return atan2f(a, b); }
__DEVICE_INLINE__ float16 atanh(const __half a) { return atanhf(a); }
__DEVICE_INLINE__ float16 cos(const __half a) { return hcos(a); }
__DEVICE_INLINE__ float16 cosh(const __half a) { return coshf(a); }
__DEVICE_INLINE__ float16 erf(const __half a) { return erff(a); }
__DEVICE_INLINE__ float16 erfc(const __half a) { return erfcf(a); }
__DEVICE_INLINE__ float16 exp(const __half a) { return hexp(a); }
__DEVICE_INLINE__ float16 exp2(const __half a) { return hexp2(a); }
__DEVICE_INLINE__ float16 exp10(const __half a) { return hexp10(a); }
__DEVICE_INLINE__ float16 expm1(const __half a) { return expm1f(a); }
__DEVICE_INLINE__ float16 tgamma(const __half a) { return tgammaf(a); }
__DEVICE_INLINE__ float16 hypot(const __half a, const __half b) { return hypotf(a, b); }
__DEVICE_INLINE__ float16 log(const __half a) { return hlog(a); }
__DEVICE_INLINE__ float16 log1p(const __half a) { return log1pf(a); }
__DEVICE_INLINE__ float16 log2(const __half a) { return hlog2(a); }
__DEVICE_INLINE__ float16 log10(const __half a) { return hlog10(a); }
__DEVICE_INLINE__ float16 lgamma(const __half a) { return lgammaf(a); }
__DEVICE_INLINE__ float16 neg(const __half a) { return __hneg(a); }
__DEVICE_INLINE__ float16 pow(const __half a, const __half b) { return powf(a, b); }
__DEVICE_INLINE__ float16 sign(const __half& a) { return __hlt(a, 0) ? -1 : 1; }
__DEVICE_INLINE__ float16 sin(const __half a) { return hsin(a); }
__DEVICE_INLINE__ float16 sinh(const __half a) { return sinhf(a); }
__DEVICE_INLINE__ float16 sqrt(const __half a) { return hsqrt(a); }
__DEVICE_INLINE__ float16 tan(const __half a) { return hsin(a) / hcos(a); }
__DEVICE_INLINE__ float16 tanh(const __half a) { return tanhf(a); }
__DEVICE_INLINE__ float16 multiplyAdd(const __half& a, const __half& b, const __half& c) {
    return __hfma(a, b, c);
}

//------------------------------------------------------------------------------
// packed variants

__DEVICE_INLINE__ float162 operator-(const __half2 a) { return __hneg2(a); }

__DEVICE_INLINE__ float162 abs(const __half2 a) { return __habs2(a); }
__DEVICE_INLINE__ float162 acos(const __half2 a) { return float162(acos(a.x), acos(a.y)); }
__DEVICE_INLINE__ float162 acosh(const __half2 a) { return float162(acosh(a.x), acosh(a.y)); }
__DEVICE_INLINE__ float162 asin(const __half2 a) { return float162(asin(a.x), asin(a.y)); }
__DEVICE_INLINE__ float162 asinh(const __half2 a) { return float162(asinh(a.x), asinh(a.y)); }
__DEVICE_INLINE__ float162 atan(const __half2 a) { return float162(atan(a.x), atan(a.y)); }
__DEVICE_INLINE__ float162 atan2(const __half2 a, const __half2 b) { return float162(atan2(a.x, b.x), atan2(a.y, b.y)); }
__DEVICE_INLINE__ float162 atanh(const __half2 a) { return float162(atanh(a.x), atanh(a.y)); }
__DEVICE_INLINE__ float162 cos(const __half2 a) { return h2cos(a); }
__DEVICE_INLINE__ float162 cosh(const __half2 a) { return float162(cosh(a.x), cosh(a.y)); }
__DEVICE_INLINE__ float162 erf(const __half2 a) { return float162(erf(a.x), erf(a.y)); }
__DEVICE_INLINE__ float162 erfc(const __half2 a) { return float162(erfc(a.x), erfc(a.y)); }
__DEVICE_INLINE__ float162 exp(const __half2 a) { return h2exp(a); }
__DEVICE_INLINE__ float162 exp2(const __half2 a) { return h2exp2(a); }
__DEVICE_INLINE__ float162 exp10(const __half2 a) { return h2exp10(a); }
__DEVICE_INLINE__ float162 expm1(const __half2 a) { return float162(expm1(a.x), expm1(a.y)); }
__DEVICE_INLINE__ float162 tgamma(const __half2 a) { return float162(tgamma(a.x), tgamma(a.y)); }
__DEVICE_INLINE__ float162 hypot(const __half2 a, const __half2 b) { return float162(hypot(a.x, b.x), hypot(a.y, b.y)); }
__DEVICE_INLINE__ float162 log(const __half2 a) { return h2log(a); }
__DEVICE_INLINE__ float162 log1p(const __half2 a) { return float162(log1p(a.x), log1p(a.y)); }
__DEVICE_INLINE__ float162 log2(const __half2 a) { return h2log2(a); }
__DEVICE_INLINE__ float162 log10(const __half2 a) { return h2log10(a); }
__DEVICE_INLINE__ float162 lgamma(const __half2 a) { return float162(lgamma(a.x), lgamma(a.y)); }
__DEVICE_INLINE__ float162 neg(const __half2 a) { return __hneg2(a); }
__DEVICE_INLINE__ float162 pow(const __half2 a, const __half2 b) { return float162(pow(a.x, b.x), pow(a.y, b.y)); }
__DEVICE_INLINE__ float162 sin(const __half2 a) { return h2sin(a); }
__DEVICE_INLINE__ float162 sinh(const __half2 a) { return float162(sinh(a.x), sinh(a.y)); }
__DEVICE_INLINE__ float162 sqrt(const __half2 a) { return h2sqrt(a); }
__DEVICE_INLINE__ float162 tan(const __half2 a) { return float162(tan(a.x), tan(a.y)); }
__DEVICE_INLINE__ float162 tanh(const __half2 a) { return float162(tanh(a.x), tanh(a.y)); }
__DEVICE_INLINE__ float162 multiplyAdd(const __half2& a, const __half2& b, const __half2& c) {
    return __hfma2(a, b, c);
}
__DEVICE_INLINE__ float162 sign(const float162& a) { return float162(sign(a.x), sign(a.y)); }

