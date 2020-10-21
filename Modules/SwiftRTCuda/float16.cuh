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
#include "cuda_macros.cuh"

typedef __half float16;
typedef __half2 float162;

//==============================================================================
// supplemental math functions
//==============================================================================
__DEVICE_INLINE__ half abs(const half a) { return __habs(a); }
__DEVICE_INLINE__ half acos(const half a) { return acosf(a); }
__DEVICE_INLINE__ half acosh(const half a) { return acoshf(a); }
__DEVICE_INLINE__ half asin(const half a) { return asinf(a); }
__DEVICE_INLINE__ half asinh(const half a) { return asinhf(a); }
__DEVICE_INLINE__ half atan(const half a) { return atanf(a); }
__DEVICE_INLINE__ half atan2(const half a, const half b) { return atan2f(a, b); }
__DEVICE_INLINE__ half atanh(const half a) { return atanhf(a); }
__DEVICE_INLINE__ half cos(const half a) { return hcos(a); }
__DEVICE_INLINE__ half cosh(const half a) { return coshf(a); }
__DEVICE_INLINE__ half erf(const half a) { return erff(a); }
__DEVICE_INLINE__ half erfc(const half a) { return erfcf(a); }
__DEVICE_INLINE__ half exp(const half a) { return hexp(a); }
__DEVICE_INLINE__ half exp2(const half a) { return hexp2(a); }
__DEVICE_INLINE__ half exp10(const half a) { return hexp10(a); }
__DEVICE_INLINE__ half expm1(const half a) { return expm1f(a); }
__DEVICE_INLINE__ half tgamma(const half a) { return tgammaf(a); }
__DEVICE_INLINE__ half hypot(const half a, const half b) { return hypotf(a, b); }
__DEVICE_INLINE__ half log(const half a) { return hlog(a); }
__DEVICE_INLINE__ half log1p(const half a) { return log1pf(a); }
__DEVICE_INLINE__ half log2(const half a) { return hlog2(a); }
__DEVICE_INLINE__ half log10(const half a) { return hlog10(a); }
__DEVICE_INLINE__ half lgamma(const half a) { return lgammaf(a); }
__DEVICE_INLINE__ half neg(const half a) { return __hneg(a); }
__DEVICE_INLINE__ half pow(const half a, const half b) { return powf(a, b); }
__DEVICE_INLINE__ half sign(const half& a) { return __hlt(a, 0) ? -1 : 1; }
__DEVICE_INLINE__ half sin(const half a) { return hsin(a); }
__DEVICE_INLINE__ half sinh(const half a) { return sinhf(a); }
__DEVICE_INLINE__ half sqrt(const half a) { return hsqrt(a); }
__DEVICE_INLINE__ half tan(const half a) { return hsin(a) / hcos(a); }
__DEVICE_INLINE__ half tanh(const half a) { return tanhf(a); }
__DEVICE_INLINE__ half multiplyAdd(const half& a, const half& b, const half& c) {
    return __hfma(a, b, c);
}

__DEVICE_INLINE__ bool isfinite(const half a) { return ::__finitef(a); }
__DEVICE_INLINE__ half max(const half a, const half b) { return a >= b ? a : b; }
__DEVICE_INLINE__ half min(const half a, const half b) { return a <= b ? a : b; }

//------------------------------------------------------------------------------
// packed variants
__DEVICE_INLINE__ half2 abs(const half2 a) { return __habs2(a); }
__DEVICE_INLINE__ half2 acos(const half2 a) { return half2(acos(a.x), acos(a.y)); }
__DEVICE_INLINE__ half2 acosh(const half2 a) { return half2(acosh(a.x), acosh(a.y)); }
__DEVICE_INLINE__ half2 asin(const half2 a) { return half2(asin(a.x), asin(a.y)); }
__DEVICE_INLINE__ half2 asinh(const half2 a) { return half2(asinh(a.x), asinh(a.y)); }
__DEVICE_INLINE__ half2 atan(const half2 a) { return half2(atan(a.x), atan(a.y)); }
__DEVICE_INLINE__ half2 atan2(const half2 a, const half2 b) { return half2(atan2(a.x, b.x), atan2(a.y, b.y)); }
__DEVICE_INLINE__ half2 atanh(const half2 a) { return half2(atanh(a.x), atanh(a.y)); }
__DEVICE_INLINE__ half2 cos(const half2 a) { return h2cos(a); }
__DEVICE_INLINE__ half2 cosh(const half2 a) { return half2(cosh(a.x), cosh(a.y)); }
__DEVICE_INLINE__ half2 erf(const half2 a) { return half2(erf(a.x), erf(a.y)); }
__DEVICE_INLINE__ half2 erfc(const half2 a) { return half2(erfc(a.x), erfc(a.y)); }
__DEVICE_INLINE__ half2 exp(const half2 a) { return h2exp(a); }
__DEVICE_INLINE__ half2 exp2(const half2 a) { return h2exp2(a); }
__DEVICE_INLINE__ half2 exp10(const half2 a) { return h2exp10(a); }
__DEVICE_INLINE__ half2 expm1(const half2 a) { return half2(expm1(a.x), expm1(a.y)); }
__DEVICE_INLINE__ half2 tgamma(const half2 a) { return half2(tgamma(a.x), tgamma(a.y)); }
__DEVICE_INLINE__ half2 hypot(const half2 a, const half2 b) { return half2(hypot(a.x, b.x), hypot(a.y, b.y)); }
__DEVICE_INLINE__ half2 log(const half2 a) { return h2log(a); }
__DEVICE_INLINE__ half2 log1p(const half2 a) { return half2(log1p(a.x), log1p(a.y)); }
__DEVICE_INLINE__ half2 log2(const half2 a) { return h2log2(a); }
__DEVICE_INLINE__ half2 log10(const half2 a) { return h2log10(a); }
__DEVICE_INLINE__ half2 lgamma(const half2 a) { return half2(lgamma(a.x), lgamma(a.y)); }
__DEVICE_INLINE__ half2 neg(const half2 a) { return __hneg2(a); }
__DEVICE_INLINE__ half2 pow(const half2 a, const half2 b) { return half2(pow(a.x, b.x), pow(a.y, b.y)); }
__DEVICE_INLINE__ half2 sin(const half2 a) { return h2sin(a); }
__DEVICE_INLINE__ half2 sinh(const half2 a) { return half2(sinh(a.x), sinh(a.y)); }
__DEVICE_INLINE__ half2 sqrt(const half2 a) { return h2sqrt(a); }
__DEVICE_INLINE__ half2 tan(const half2 a) { return half2(tan(a.x), tan(a.y)); }
__DEVICE_INLINE__ half2 tanh(const half2 a) { return half2(tanh(a.x), tanh(a.y)); }
__DEVICE_INLINE__ half2 multiplyAdd(const half2& a, const half2& b, const half2& c) {
    return __hfma2(a, b, c);
}
__DEVICE_INLINE__ half2 sign(const half2& a) { return half2(sign(a.x), sign(a.y)); }


