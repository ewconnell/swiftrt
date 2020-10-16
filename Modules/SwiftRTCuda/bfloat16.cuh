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
#include <cuda_bf16.h>
#include "srt_types.cuh"

struct bfloat16 : public __nv_bfloat16 {
    __HOSTDEVICE_INLINE__ bfloat16() : __nv_bfloat16(float(0)) {}
    __HOSTDEVICE_INLINE__ bfloat16(__nv_bfloat16 v) : __nv_bfloat16(v) {}
    __HOSTDEVICE_INLINE__ bfloat16(float v) : __nv_bfloat16(__float2bfloat16(v)) {}
    __HOSTDEVICE_INLINE__ operator float() const { return __bfloat162float(*this); }
};

struct bfloat162 : public __nv_bfloat162 {
    __HOSTDEVICE_INLINE__ bfloat162() : __nv_bfloat162(float(0), float(0)) {}
    __HOSTDEVICE_INLINE__ bfloat162(__nv_bfloat162 v) : __nv_bfloat162(v) {}
    __HOSTDEVICE_INLINE__ bfloat162(__nv_bfloat16 __x, __nv_bfloat16 __y) : __nv_bfloat162(__x, __y) {}
    __HOSTDEVICE_INLINE__ bfloat162(float2 v) : __nv_bfloat162(init_float22bfloat162_rn(v)) {}
    __HOSTDEVICE_INLINE__ bfloat162(float v) : __nv_bfloat162(init_float22bfloat162_rn(make_float2(v, v))) {}
    __HOSTDEVICE_INLINE__ bfloat162(float x, float y) : __nv_bfloat162(init_float22bfloat162_rn(make_float2(x, y))) {}
    __HOSTDEVICE_INLINE__ operator float2() const { return init_bfloat1622float2(*this); }

    __HOSTDEVICE_INLINE__ static __nv_bfloat162 init_float22bfloat162_rn(float2 v) {
        return __nv_bfloat162(__float2bfloat16(v.x), __float2bfloat16(v.y));
    }
    
    __HOSTDEVICE_INLINE__ static float2 init_bfloat1622float2(__nv_bfloat162 v) {
        return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
    }    
};

//==============================================================================
// supplemental functions
//==============================================================================

#if (__CUDA_ARCH__ < 800)
__DEVICE_INLINE__ bfloat162 operator+(const bfloat162& a, const bfloat162& b) {
    return bfloat162(a.x + b.x, a.y + b.y);
}

__DEVICE_INLINE__ bfloat16 abs(const __nv_bfloat16 a) { return abs(float(a)); }
__DEVICE_INLINE__ bfloat16 cos(const __nv_bfloat16 a) { return cosf(a); }
__DEVICE_INLINE__ bfloat16 exp(const __nv_bfloat16 a) { return expf(a); }
__DEVICE_INLINE__ bfloat16 exp2(const __nv_bfloat16 a) { return exp2f(a); }
__DEVICE_INLINE__ bfloat16 exp10(const __nv_bfloat16 a) { return exp10f(a); }
__DEVICE_INLINE__ bfloat16 log(const __nv_bfloat16 a) { return logf(a); }

__DEVICE_INLINE__ bfloat16 log2(const __nv_bfloat16 a) { return log2f(a); }
__DEVICE_INLINE__ bfloat16 log10(const __nv_bfloat16 a) { return log10f(a); }
__DEVICE_INLINE__ bfloat16 neg(const __nv_bfloat16 a) { return -a; }
__DEVICE_INLINE__ bfloat16 sign(const __nv_bfloat16& a) { return float(a) < 0 ? -1 : 1; }
__DEVICE_INLINE__ bfloat16 sin(const __nv_bfloat16 a) { return sinf(a); }
__DEVICE_INLINE__ bfloat16 sqrt(const __nv_bfloat16 a) { return sqrtf(a); }
__DEVICE_INLINE__ bfloat16 tan(const __nv_bfloat16 a) { return tanf(a); }
__DEVICE_INLINE__ bfloat16 multiplyAdd(const __nv_bfloat16& a, const __nv_bfloat16& b, const __nv_bfloat16& c) {
    return a * b + c;
}

// packed
__DEVICE_INLINE__ bfloat162 cos(const __nv_bfloat162 a) { return bfloat162(cos(a.x), cos(a.y)); }
__DEVICE_INLINE__ bfloat162 exp(const __nv_bfloat162 a) { return bfloat162(exp(a.x), exp(a.y)); }
__DEVICE_INLINE__ bfloat162 exp2(const __nv_bfloat162 a) { return bfloat162(exp2(a.x), exp2(a.y)); }
__DEVICE_INLINE__ bfloat162 exp10(const __nv_bfloat162 a) { return bfloat162(exp10(a.x), exp10(a.y)); }
__DEVICE_INLINE__ bfloat162 log(const __nv_bfloat162 a) { return bfloat162(log(a.x), log(a.y)); }
__DEVICE_INLINE__ bfloat162 log2(const __nv_bfloat162 a) { return bfloat162(log2(a.x), log2(a.y)); }
__DEVICE_INLINE__ bfloat162 log10(const __nv_bfloat162 a) { return bfloat162(log10(a.x), log10(a.y)); }
__DEVICE_INLINE__ bfloat162 neg(const __nv_bfloat162 a) { return bfloat162(-a.x, -a.y); }
__DEVICE_INLINE__ bfloat162 sin(const __nv_bfloat162 a) { return bfloat162(sin(a.x), sin(a.y)); }
__DEVICE_INLINE__ bfloat162 sqrt(const __nv_bfloat162 a) { return bfloat162(sqrt(a.x), sqrt(a.y)); }
__DEVICE_INLINE__ bfloat162 abs(const __nv_bfloat162 a) { return bfloat162(abs(a.x), abs(a.y)); }
__DEVICE_INLINE__ bfloat162 multiplyAdd(const __nv_bfloat162& a, const __nv_bfloat162& b, const __nv_bfloat162& c) {
    return bfloat162(a.x * b.x + c.x, a.y * b.y + c.y);
}

#else

//------------------------------------------------------------------------------
// native variants
__DEVICE_INLINE__ bfloat16 abs(const __nv_bfloat16 a) { return __habs(a); }
__DEVICE_INLINE__ bfloat16 cos(const __nv_bfloat16 a) { return hcos(a); }
__DEVICE_INLINE__ bfloat16 exp(const __nv_bfloat16 a) { return hexp(a); }
__DEVICE_INLINE__ bfloat16 exp2(const __nv_bfloat16 a) { return hexp2(a); }
__DEVICE_INLINE__ bfloat16 exp10(const __nv_bfloat16 a) { return hexp10(a); }
__DEVICE_INLINE__ bfloat16 log(const __nv_bfloat16 a) { return hlog(a); }
__DEVICE_INLINE__ bfloat16 log2(const __nv_bfloat16 a) { return hlog2(a); }
__DEVICE_INLINE__ bfloat16 log10(const __nv_bfloat16 a) { return hlog10(a); }
__DEVICE_INLINE__ bfloat16 neg(const __nv_bfloat16 a) { return __hneg(a); }
__DEVICE_INLINE__ bfloat16 sign(const __nv_bfloat16& a) { return __hlt(a, 0) ? -1 : 1; }
__DEVICE_INLINE__ bfloat16 sin(const __nv_bfloat16 a) { return hsin(a); }
__DEVICE_INLINE__ bfloat16 sqrt(const __nv_bfloat16 a) { return hsqrt(a); }
__DEVICE_INLINE__ bfloat16 tan(const __nv_bfloat16 a) { return hsin(a) / hcos(a); }
__DEVICE_INLINE__ bfloat16 multiplyAdd(const __nv_bfloat16& a, const __nv_bfloat16& b, const __nv_bfloat16& c) {
    return __hfma(a, b, c);
}

// packed
__DEVICE_INLINE__ bfloat162 cos(const __nv_bfloat162 a) { return h2cos(a); }
__DEVICE_INLINE__ bfloat162 exp(const __nv_bfloat162 a) { return h2exp(a); }
__DEVICE_INLINE__ bfloat162 exp2(const __nv_bfloat162 a) { return h2exp2(a); }
__DEVICE_INLINE__ bfloat162 exp10(const __nv_bfloat162 a) { return h2exp10(a); }
__DEVICE_INLINE__ bfloat162 log(const __nv_bfloat162 a) { return h2log(a); }
__DEVICE_INLINE__ bfloat162 log2(const __nv_bfloat162 a) { return h2log2(a); }
__DEVICE_INLINE__ bfloat162 log10(const __nv_bfloat162 a) { return h2log10(a); }
__DEVICE_INLINE__ bfloat162 neg(const __nv_bfloat162 a) { return __hneg2(a); }
__DEVICE_INLINE__ bfloat162 sin(const __nv_bfloat162 a) { return h2sin(a); }
__DEVICE_INLINE__ bfloat162 sqrt(const __nv_bfloat162 a) { return h2sqrt(a); }
__DEVICE_INLINE__ bfloat162 abs(const __nv_bfloat162 a) { return __habs2(a); }
__DEVICE_INLINE__ bfloat162 multiplyAdd(const __nv_bfloat162& a, const __nv_bfloat162& b, const __nv_bfloat162& c) {
    return __hfma2(a, b, c);
}

#endif

//------------------------------------------------------------------------------
// emulated variants

__DEVICE_INLINE__ bfloat16 acos(const __nv_bfloat16 a) { return acosf(a); }
__DEVICE_INLINE__ bfloat16 acosh(const __nv_bfloat16 a) { return acoshf(a); }
__DEVICE_INLINE__ bfloat16 asin(const __nv_bfloat16 a) { return asinf(a); }
__DEVICE_INLINE__ bfloat16 asinh(const __nv_bfloat16 a) { return asinhf(a); }
__DEVICE_INLINE__ bfloat16 atan(const __nv_bfloat16 a) { return atanf(a); }
__DEVICE_INLINE__ bfloat16 atan2(const __nv_bfloat16 a, const __nv_bfloat16 b) { return atan2f(a, b); }
__DEVICE_INLINE__ bfloat16 atanh(const __nv_bfloat16 a) { return atanhf(a); }
__DEVICE_INLINE__ bfloat16 cosh(const __nv_bfloat16 a) { return coshf(a); }
__DEVICE_INLINE__ bfloat16 erf(const __nv_bfloat16 a) { return erff(a); }
__DEVICE_INLINE__ bfloat16 erfc(const __nv_bfloat16 a) { return erfcf(a); }
__DEVICE_INLINE__ bfloat16 expm1(const __nv_bfloat16 a) { return expm1f(a); }
__DEVICE_INLINE__ bfloat16 tgamma(const __nv_bfloat16 a) { return tgammaf(a); }
__DEVICE_INLINE__ bfloat16 hypot(const __nv_bfloat16 a, const __nv_bfloat16 b) { return hypotf(a, b); }
__DEVICE_INLINE__ bfloat16 log1p(const __nv_bfloat16 a) { return log1pf(a); }
__DEVICE_INLINE__ bfloat16 lgamma(const __nv_bfloat16 a) { return lgammaf(a); }
__DEVICE_INLINE__ bfloat16 pow(const __nv_bfloat16 a, const __nv_bfloat16 b) { return powf(a, b); }
__DEVICE_INLINE__ bfloat16 sinh(const __nv_bfloat16 a) { return sinhf(a); }
__DEVICE_INLINE__ bfloat16 tanh(const __nv_bfloat16 a) { return tanhf(a); }

//------------------------------------------------------------------------------
// packed variants

__DEVICE_INLINE__ bfloat162 acos(const __nv_bfloat162 a) { return bfloat162(acos(a.x), acos(a.y)); }
__DEVICE_INLINE__ bfloat162 acosh(const __nv_bfloat162 a) { return bfloat162(acosh(a.x), acosh(a.y)); }
__DEVICE_INLINE__ bfloat162 asin(const __nv_bfloat162 a) { return bfloat162(asin(a.x), asin(a.y)); }
__DEVICE_INLINE__ bfloat162 asinh(const __nv_bfloat162 a) { return bfloat162(asinh(a.x), asinh(a.y)); }
__DEVICE_INLINE__ bfloat162 atan(const __nv_bfloat162 a) { return bfloat162(atan(a.x), atan(a.y)); }
__DEVICE_INLINE__ bfloat162 atan2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return bfloat162(atan2(a.x, b.x), atan2(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 atanh(const __nv_bfloat162 a) { return bfloat162(atanh(a.x), atanh(a.y)); }
__DEVICE_INLINE__ bfloat162 cosh(const __nv_bfloat162 a) { return bfloat162(cosh(a.x), cosh(a.y)); }
__DEVICE_INLINE__ bfloat162 erf(const __nv_bfloat162 a) { return bfloat162(erf(a.x), erf(a.y)); }
__DEVICE_INLINE__ bfloat162 erfc(const __nv_bfloat162 a) { return bfloat162(erfc(a.x), erfc(a.y)); }
__DEVICE_INLINE__ bfloat162 expm1(const __nv_bfloat162 a) { return bfloat162(expm1(a.x), expm1(a.y)); }
__DEVICE_INLINE__ bfloat162 tgamma(const __nv_bfloat162 a) { return bfloat162(tgamma(a.x), tgamma(a.y)); }
__DEVICE_INLINE__ bfloat162 hypot(const __nv_bfloat162 a, const __nv_bfloat162 b) { return bfloat162(hypot(a.x, b.x), hypot(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 log1p(const __nv_bfloat162 a) { return bfloat162(log1p(a.x), log1p(a.y)); }
__DEVICE_INLINE__ bfloat162 lgamma(const __nv_bfloat162 a) { return bfloat162(lgamma(a.x), lgamma(a.y)); }
__DEVICE_INLINE__ bfloat162 pow(const __nv_bfloat162 a, const __nv_bfloat162 b) { return bfloat162(pow(a.x, b.x), pow(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 sign(const __nv_bfloat162& a) { return bfloat162(sign(a.x), sign(a.y)); }
__DEVICE_INLINE__ bfloat162 sinh(const __nv_bfloat162 a) { return bfloat162(sinh(a.x), sinh(a.y)); }
__DEVICE_INLINE__ bfloat162 tan(const __nv_bfloat162 a) { return bfloat162(tan(a.x), tan(a.y)); }
__DEVICE_INLINE__ bfloat162 tanh(const __nv_bfloat162 a) { return bfloat162(tanh(a.x), tanh(a.y)); }
__DEVICE_INLINE__ bfloat162 squared(const __nv_bfloat162& a) {
    return bfloat162(a.x * a.x, a.y * a.y);
}
