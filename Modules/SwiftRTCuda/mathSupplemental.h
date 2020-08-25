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
#ifndef mathSupplemental_h
#define mathSupplemental_h

#include <cuda_fp16.h>
#include <cuda_bf16.h>

//==============================================================================
// supplemental function delegating macros
//==============================================================================

#define NATIVE_FLOAT16(func, native) \
__device__ inline __half func(const __half& a) { return native(a); }

#define NATIVE_FLOAT162(func, native) \
__device__ inline __half2 func(const __half2& a) { return native(a); }

#if (__CUDA_ARCH__ < 800)
#define NATIVE_BFLOAT16(func, native) \
    __device__ inline __nv_bfloat16 func(const __nv_bfloat16& a) \
    { return func(float(a)); }
#else
#define NATIVE_BFLOAT16(func, native) \
    __device__ inline __nv_bfloat16 func(const __nv_bfloat16& a) { return native(a); }
#endif

#if (__CUDA_ARCH__ < 800)
#define NATIVE_BFLOAT162(func, native) \
    __device__ inline __nv_bfloat162 func(const __nv_bfloat162& a) \
    { return __nv_bfloat162(func(float(a.x)), func(float(a.y))); }
#else
#define NATIVE_BFLOAT162(func, native) \
    __device__ inline __nv_bfloat162 func(const __nv_bfloat162& a) { return native(a); }
#endif

//------------------------------------------------------------------------------
// Promotes the type to a float, does the op, then back to the type. This
// is used for ops that do not natively support half or bfloat types
#define PROMOTED_FLOAT16(func) \
__device__ inline __half func(const __half& a) { return func(float(a)); }

#define PROMOTED_FLOAT162(func) \
__device__ inline __half2 func(const __half2& a) { return __half2(func(a.x), func(a.y)); }

#define PROMOTED_BFLOAT16(func) \
__device__ inline __nv_bfloat16 func(const __nv_bfloat16& a) { return func(float(a)); }

#define PROMOTED_BFLOAT162(func) \
__device__ inline __nv_bfloat162 func(const __nv_bfloat162& a) \
    { return __nv_bfloat162(func(float(a.x)), func(float(a.y))); }

//------------------------------------------------------------------------------
// Promotes the type to a float, does the op, then back to the type. This
// is used for ops that do not natively support half or bfloat types
#define PROMOTED2_FLOAT16(func) \
__device__ inline __half func(const __half& a, const __half& b) \
    { return func(float(a), float(b)); }

#define PROMOTED2_FLOAT162(func) \
__device__ inline __half2 func(const __half2& a, const __half2& b) \
    { return __half2(func(a.x, b.x), func(a.y, b.y)); }

#define PROMOTED2_BFLOAT16(func) \
__device__ inline __nv_bfloat16 func(const __nv_bfloat16& a, const __nv_bfloat16& b) \
    { return func(float(a), float(b)); }

#define PROMOTED2_BFLOAT162(func) \
__device__ inline __nv_bfloat162 func(const __nv_bfloat162& a, const __nv_bfloat162& b) \
    { return __nv_bfloat162(func(float(a.x), float(b.x)), func(float(a.y), float(b.y))); }

//==============================================================================
// supplemental functions
//==============================================================================

// abs
NATIVE_FLOAT16(abs, __habs)
NATIVE_FLOAT162(abs, __habs2)
NATIVE_BFLOAT16(abs, __habs)
NATIVE_BFLOAT162(abs, __habs2)

// acos
PROMOTED_FLOAT16(acos)
PROMOTED_FLOAT162(acos)
PROMOTED_BFLOAT16(acos)
PROMOTED_BFLOAT162(acos)

// acosh
PROMOTED_FLOAT16(acosh)
PROMOTED_FLOAT162(acosh)
PROMOTED_BFLOAT16(acosh)
PROMOTED_BFLOAT162(acosh)

// asin
PROMOTED_FLOAT16(asin)
PROMOTED_FLOAT162(asin)
PROMOTED_BFLOAT16(asin)
PROMOTED_BFLOAT162(asin)

// asinh
PROMOTED_FLOAT16(asinh)
PROMOTED_FLOAT162(asinh)
PROMOTED_BFLOAT16(asinh)
PROMOTED_BFLOAT162(asinh)

// atan
PROMOTED_FLOAT16(atan)
PROMOTED_FLOAT162(atan)
PROMOTED_BFLOAT16(atan)
PROMOTED_BFLOAT162(atan)

// atan2
PROMOTED2_FLOAT16(atan2)
PROMOTED2_FLOAT162(atan2)
PROMOTED2_BFLOAT16(atan2)
PROMOTED2_BFLOAT162(atan2)

// atanh
PROMOTED_FLOAT16(atanh)
PROMOTED_FLOAT162(atanh)
PROMOTED_BFLOAT16(atanh)
PROMOTED_BFLOAT162(atanh)

// cos
NATIVE_FLOAT16(cos, hcos)
NATIVE_FLOAT162(cos, h2cos)
NATIVE_BFLOAT16(cos, hcos)
NATIVE_BFLOAT162(cos, h2cos)

// cosh
PROMOTED_FLOAT16(cosh)
PROMOTED_FLOAT162(cosh)
PROMOTED_BFLOAT16(cosh)
PROMOTED_BFLOAT162(cosh)

// erf
PROMOTED_FLOAT16(erf)
PROMOTED_FLOAT162(erf)
PROMOTED_BFLOAT16(erf)
PROMOTED_BFLOAT162(erf)

// erfc
PROMOTED_FLOAT16(erfc)
PROMOTED_FLOAT162(erfc)
PROMOTED_BFLOAT16(erfc)
PROMOTED_BFLOAT162(erfc)

// exp
NATIVE_FLOAT16(exp, hexp)
NATIVE_FLOAT162(exp, h2exp)
NATIVE_BFLOAT16(exp, hexp)
NATIVE_BFLOAT162(exp, h2exp)

// exp2
NATIVE_FLOAT16(exp2, hexp2)
NATIVE_FLOAT162(exp2, h2exp2)
NATIVE_BFLOAT16(exp2, hexp2)
NATIVE_BFLOAT162(exp2, h2exp2)

// exp10
NATIVE_FLOAT16(exp10, hexp10)
NATIVE_FLOAT162(exp10, h2exp10)
NATIVE_BFLOAT16(exp10, hexp10)
NATIVE_BFLOAT162(exp10, h2exp10)

// expm1
PROMOTED_FLOAT16(expm1)
PROMOTED_FLOAT162(expm1)
PROMOTED_BFLOAT16(expm1)
PROMOTED_BFLOAT162(expm1)

// tgamma
PROMOTED_FLOAT16(tgamma)
PROMOTED_FLOAT162(tgamma)
PROMOTED_BFLOAT16(tgamma)
PROMOTED_BFLOAT162(tgamma)

// hypot
PROMOTED2_FLOAT16(hypot)
PROMOTED2_FLOAT162(hypot)
PROMOTED2_BFLOAT16(hypot)
PROMOTED2_BFLOAT162(hypot)

// log
NATIVE_FLOAT16(log, hlog)
NATIVE_FLOAT162(log, h2log)
NATIVE_BFLOAT16(log, hlog)
NATIVE_BFLOAT162(log, h2log)

// log1p
PROMOTED_FLOAT16(log1p)
PROMOTED_FLOAT162(log1p)
PROMOTED_BFLOAT16(log1p)
PROMOTED_BFLOAT162(log1p)

// log2
NATIVE_FLOAT16(log2, hlog2)
NATIVE_FLOAT162(log2, h2log2)
NATIVE_BFLOAT16(log2, hlog2)
NATIVE_BFLOAT162(log2, h2log2)

// log10
NATIVE_FLOAT16(log10, hlog10)
NATIVE_FLOAT162(log10, h2log10)
NATIVE_BFLOAT16(log10, hlog10)
NATIVE_BFLOAT162(log10, h2log10)

// lgamma
PROMOTED_FLOAT16(lgamma)
PROMOTED_FLOAT162(lgamma)
PROMOTED_BFLOAT16(lgamma)
PROMOTED_BFLOAT162(lgamma)

// neg
template<typename T>
__device__ inline T neg(const T& a) { return -a; }

// pow
PROMOTED2_FLOAT16(pow)
PROMOTED2_FLOAT162(pow)
PROMOTED2_BFLOAT16(pow)
PROMOTED2_BFLOAT162(pow)

//==============================================================================
// supplemental custom functions
//==============================================================================

//------------------------------------------------------------------------------
// sigmoid Float16
template<typename T>
__device__ inline T sigmoid(const T& a) { return T(1) / (T(1) + exp(-a)); }

__device__ inline __half2 sigmoid(const __half2& a) {
    const __half2 one = __half2(1,1);
    return one / (one + exp(-a));
}

// sigmoid BFloat16
#if (__CUDA_ARCH__ < 800)
__device__ inline __nv_bfloat16 sigmoid(const __nv_bfloat16& a) {
    return sigmoid(float(a));
}

__device__ inline __nv_bfloat162 sigmoid(const __nv_bfloat162& a) {
    return __nv_bfloat162(sigmoid(float(a.x)), sigmoid(float(a.y)));
}
#else

#endif

//==============================================================================
#endif // mathSupplemental_h