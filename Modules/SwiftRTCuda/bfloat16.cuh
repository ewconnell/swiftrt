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
#include "cuda_macros.cuh"

typedef __nv_bfloat16 bfloat16;
typedef __nv_bfloat162 bfloat162;

//==============================================================================
// supplemental functions
//==============================================================================

#if (__CUDA_ARCH__ < 800)
__DEVICE_INLINE__ bfloat16 abs(const bfloat16 a) { return abs(float(a)); }
__DEVICE_INLINE__ bfloat16 cos(const bfloat16 a) { return cosf(a); }
__DEVICE_INLINE__ bfloat16 exp(const bfloat16 a) { return expf(a); }
__DEVICE_INLINE__ bfloat16 exp2(const bfloat16 a) { return exp2f(a); }
__DEVICE_INLINE__ bfloat16 exp10(const bfloat16 a) { return exp10f(a); }
__DEVICE_INLINE__ bfloat16 log(const bfloat16 a) { return logf(a); }

__DEVICE_INLINE__ bfloat16 log2(const bfloat16 a) { return log2f(a); }
__DEVICE_INLINE__ bfloat16 log10(const bfloat16 a) { return log10f(a); }
__DEVICE_INLINE__ bfloat16 neg(const bfloat16 a) { return -a; }
__DEVICE_INLINE__ bfloat16 sign(const bfloat16& a) { return a < __float2bfloat16(0) ? __float2bfloat16(-1) : __float2bfloat16(1); }
__DEVICE_INLINE__ bfloat16 sin(const bfloat16 a) { return sinf(a); }
__DEVICE_INLINE__ bfloat16 sqrt(const bfloat16 a) { return sqrtf(a); }
__DEVICE_INLINE__ bfloat16 tan(const bfloat16 a) { return tanf(a); }
__DEVICE_INLINE__ bfloat16 multiplyAdd(const bfloat16& a, const bfloat16& b, const bfloat16& c) {
    return a * b + c;
}

// packed
__DEVICE_INLINE__ bfloat162 cos(const bfloat162 a) { return bfloat162(cos(a.x), cos(a.y)); }
__DEVICE_INLINE__ bfloat162 exp(const bfloat162 a) { return bfloat162(exp(a.x), exp(a.y)); }
__DEVICE_INLINE__ bfloat162 exp2(const bfloat162 a) { return bfloat162(exp2(a.x), exp2(a.y)); }
__DEVICE_INLINE__ bfloat162 exp10(const bfloat162 a) { return bfloat162(exp10(a.x), exp10(a.y)); }
__DEVICE_INLINE__ bfloat162 log(const bfloat162 a) { return bfloat162(log(a.x), log(a.y)); }
__DEVICE_INLINE__ bfloat162 log2(const bfloat162 a) { return bfloat162(log2(a.x), log2(a.y)); }
__DEVICE_INLINE__ bfloat162 log10(const bfloat162 a) { return bfloat162(log10(a.x), log10(a.y)); }
__DEVICE_INLINE__ bfloat162 neg(const bfloat162 a) { return bfloat162(-a.x, -a.y); }
__DEVICE_INLINE__ bfloat162 sin(const bfloat162 a) { return bfloat162(sin(a.x), sin(a.y)); }
__DEVICE_INLINE__ bfloat162 sqrt(const bfloat162 a) { return bfloat162(sqrt(a.x), sqrt(a.y)); }
__DEVICE_INLINE__ bfloat162 abs(const bfloat162 a) { return bfloat162(abs(a.x), abs(a.y)); }

//------------------------------------------------------------------------------
// these are not defined as operators because of cuda header conflicts
__DEVICE_INLINE__ bfloat162 add(const bfloat162& a, const bfloat162& b) {
    return bfloat162(a.x + b.x, a.y + b.y);
}

__DEVICE_INLINE__ bfloat162 subtract(const bfloat162& a, const bfloat162& b) {
    return bfloat162(a.x - b.x, a.y - b.y);
}

__DEVICE_INLINE__ bfloat162 multiply(const bfloat162& a, const bfloat162& b) {
    return bfloat162(a.x * b.x, a.y * b.y);
}

__DEVICE_INLINE__ bfloat162 multiplyAdd(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
    return bfloat162(a.x * b.x + c.x, a.y * b.y + c.y);
}

__DEVICE_INLINE__ bfloat162 divide(const bfloat162& a, const bfloat162& b) {
    return bfloat162(a.x / b.x, a.y / b.y);
}

#else

//------------------------------------------------------------------------------
// native variants
__DEVICE_INLINE__ bfloat16 abs(const bfloat16 a) { return __habs(a); }
__DEVICE_INLINE__ bfloat16 cos(const bfloat16 a) { return hcos(a); }
__DEVICE_INLINE__ bfloat16 exp(const bfloat16 a) { return hexp(a); }
__DEVICE_INLINE__ bfloat16 exp2(const bfloat16 a) { return hexp2(a); }
__DEVICE_INLINE__ bfloat16 exp10(const bfloat16 a) { return hexp10(a); }
__DEVICE_INLINE__ bfloat16 log(const bfloat16 a) { return hlog(a); }
__DEVICE_INLINE__ bfloat16 log2(const bfloat16 a) { return hlog2(a); }
__DEVICE_INLINE__ bfloat16 log10(const bfloat16 a) { return hlog10(a); }
__DEVICE_INLINE__ bfloat16 neg(const bfloat16 a) { return __hneg(a); }
__DEVICE_INLINE__ bfloat16 sign(const bfloat16& a) { return __hlt(a, 0) ? -1 : 1; }
__DEVICE_INLINE__ bfloat16 sin(const bfloat16 a) { return hsin(a); }
__DEVICE_INLINE__ bfloat16 sqrt(const bfloat16 a) { return hsqrt(a); }
__DEVICE_INLINE__ bfloat16 tan(const bfloat16 a) { return hsin(a) / hcos(a); }
__DEVICE_INLINE__ bfloat16 multiplyAdd(const bfloat16& a, const bfloat16& b, const bfloat16& c) {
    return __hfma(a, b, c);
}

// packed
__DEVICE_INLINE__ bfloat162 cos(const bfloat162 a) { return h2cos(a); }
__DEVICE_INLINE__ bfloat162 exp(const bfloat162 a) { return h2exp(a); }
__DEVICE_INLINE__ bfloat162 exp2(const bfloat162 a) { return h2exp2(a); }
__DEVICE_INLINE__ bfloat162 exp10(const bfloat162 a) { return h2exp10(a); }
__DEVICE_INLINE__ bfloat162 log(const bfloat162 a) { return h2log(a); }
__DEVICE_INLINE__ bfloat162 log2(const bfloat162 a) { return h2log2(a); }
__DEVICE_INLINE__ bfloat162 log10(const bfloat162 a) { return h2log10(a); }
__DEVICE_INLINE__ bfloat162 neg(const bfloat162 a) { return __hneg2(a); }
__DEVICE_INLINE__ bfloat162 sin(const bfloat162 a) { return h2sin(a); }
__DEVICE_INLINE__ bfloat162 sqrt(const bfloat162 a) { return h2sqrt(a); }
__DEVICE_INLINE__ bfloat162 abs(const bfloat162 a) { return __habs2(a); }

//------------------------------------------------------------------------------
// these are not defined as operators because of cuda header conflicts
__DEVICE_INLINE__ bfloat162 add(const bfloat162& a, const bfloat162& b) {
    return __hadd2(a, b);
}

__DEVICE_INLINE__ bfloat162 subtract(const bfloat162& a, const bfloat162& b) {
    return __hsub2(a, b);
}

__DEVICE_INLINE__ bfloat162 multiply(const bfloat162& a, const bfloat162& b) {
    return __hmul2(a, b);
}

__DEVICE_INLINE__ bfloat162 multiplyAdd(const bfloat162& a, const bfloat162& b, const bfloat162& c) {
    return __hfma2(a, b, c);
}

__DEVICE_INLINE__ bfloat162 divide(const bfloat162& a, const bfloat162& b) {
    return __h2div(a, b);
}

#endif

//------------------------------------------------------------------------------
// emulated variants

__DEVICE_INLINE__ bfloat16 acos(const bfloat16 a) { return acosf(a); }
__DEVICE_INLINE__ bfloat16 acosh(const bfloat16 a) { return acoshf(a); }
__DEVICE_INLINE__ bfloat16 asin(const bfloat16 a) { return asinf(a); }
__DEVICE_INLINE__ bfloat16 asinh(const bfloat16 a) { return asinhf(a); }
__DEVICE_INLINE__ bfloat16 atan(const bfloat16 a) { return atanf(a); }
__DEVICE_INLINE__ bfloat16 atan2(const bfloat16 a, const bfloat16 b) { return atan2f(a, b); }
__DEVICE_INLINE__ bfloat16 atanh(const bfloat16 a) { return atanhf(a); }
__DEVICE_INLINE__ bfloat16 cosh(const bfloat16 a) { return coshf(a); }
__DEVICE_INLINE__ bfloat16 erf(const bfloat16 a) { return erff(a); }
__DEVICE_INLINE__ bfloat16 erfc(const bfloat16 a) { return erfcf(a); }
__DEVICE_INLINE__ bfloat16 expm1(const bfloat16 a) { return expm1f(a); }
__DEVICE_INLINE__ bfloat16 tgamma(const bfloat16 a) { return tgammaf(a); }
__DEVICE_INLINE__ bfloat16 hypot(const bfloat16 a, const bfloat16 b) { return hypotf(a, b); }
__DEVICE_INLINE__ bfloat16 log1p(const bfloat16 a) { return log1pf(a); }
__DEVICE_INLINE__ bfloat16 lgamma(const bfloat16 a) { return lgammaf(a); }
__DEVICE_INLINE__ bfloat16 pow(const bfloat16 a, const bfloat16 b) { return powf(a, b); }
__DEVICE_INLINE__ bfloat16 sinh(const bfloat16 a) { return sinhf(a); }
__DEVICE_INLINE__ bfloat16 tanh(const bfloat16 a) { return tanhf(a); }

//------------------------------------------------------------------------------
// packed variants

__DEVICE_INLINE__ bfloat162 acos(const bfloat162 a) { return bfloat162(acos(a.x), acos(a.y)); }
__DEVICE_INLINE__ bfloat162 acosh(const bfloat162 a) { return bfloat162(acosh(a.x), acosh(a.y)); }
__DEVICE_INLINE__ bfloat162 asin(const bfloat162 a) { return bfloat162(asin(a.x), asin(a.y)); }
__DEVICE_INLINE__ bfloat162 asinh(const bfloat162 a) { return bfloat162(asinh(a.x), asinh(a.y)); }
__DEVICE_INLINE__ bfloat162 atan(const bfloat162 a) { return bfloat162(atan(a.x), atan(a.y)); }
__DEVICE_INLINE__ bfloat162 atan2(const bfloat162 a, const bfloat162 b) { return bfloat162(atan2(a.x, b.x), atan2(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 atanh(const bfloat162 a) { return bfloat162(atanh(a.x), atanh(a.y)); }
__DEVICE_INLINE__ bfloat162 cosh(const bfloat162 a) { return bfloat162(cosh(a.x), cosh(a.y)); }
__DEVICE_INLINE__ bfloat162 erf(const bfloat162 a) { return bfloat162(erf(a.x), erf(a.y)); }
__DEVICE_INLINE__ bfloat162 erfc(const bfloat162 a) { return bfloat162(erfc(a.x), erfc(a.y)); }
__DEVICE_INLINE__ bfloat162 expm1(const bfloat162 a) { return bfloat162(expm1(a.x), expm1(a.y)); }
__DEVICE_INLINE__ bfloat162 tgamma(const bfloat162 a) { return bfloat162(tgamma(a.x), tgamma(a.y)); }
__DEVICE_INLINE__ bfloat162 hypot(const bfloat162 a, const bfloat162 b) { return bfloat162(hypot(a.x, b.x), hypot(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 log1p(const bfloat162 a) { return bfloat162(log1p(a.x), log1p(a.y)); }
__DEVICE_INLINE__ bfloat162 lgamma(const bfloat162 a) { return bfloat162(lgamma(a.x), lgamma(a.y)); }
__DEVICE_INLINE__ bfloat162 pow(const bfloat162 a, const bfloat162 b) { return bfloat162(pow(a.x, b.x), pow(a.y, b.y)); }
__DEVICE_INLINE__ bfloat162 sign(const bfloat162& a) { return bfloat162(sign(a.x), sign(a.y)); }
__DEVICE_INLINE__ bfloat162 sinh(const bfloat162 a) { return bfloat162(sinh(a.x), sinh(a.y)); }
__DEVICE_INLINE__ bfloat162 tan(const bfloat162 a) { return bfloat162(tan(a.x), tan(a.y)); }
__DEVICE_INLINE__ bfloat162 tanh(const bfloat162 a) { return bfloat162(tanh(a.x), tanh(a.y)); }
__DEVICE_INLINE__ bfloat162 squared(const bfloat162& a) {
    return bfloat162(a.x * a.x, a.y * a.y);
}
