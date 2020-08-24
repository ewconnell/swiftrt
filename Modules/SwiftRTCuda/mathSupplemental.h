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

#define PROMOTED_FLOAT16(func) \
__device__ inline __half func(const __half& a) { return func(float(a)); }

#define PROMOTED_FLOAT162(func) \
__device__ inline __half2 func(const __half2& a) { return __half2(func(a.x), func(a.y)); }

#define PROMOTED_BFLOAT16(func) \
    __device__ inline __nv_bfloat16 func(const __nv_bfloat16& a) { return func(float(a)); }

#define PROMOTED_BFLOAT162(func) \
    __device__ inline __nv_bfloat162 func(const __nv_bfloat162& a) \
    { return __nv_bfloat162(func(float(a.x)), func(float(a.y))); }

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

// exp
NATIVE_FLOAT16(exp, hexp)
NATIVE_FLOAT162(exp, h2exp)
NATIVE_BFLOAT16(exp, hexp)
NATIVE_BFLOAT162(exp, h2exp)

// cos
NATIVE_FLOAT16(cos, hcos)
NATIVE_FLOAT162(cos, h2cos)
NATIVE_BFLOAT16(cos, hcos)
NATIVE_BFLOAT162(cos, h2cos)

//==============================================================================
// supplemental custom functions
//==============================================================================

// __device__ static inline __half operator+(const __half& a, const __half& b) {
//     return __hadd(a, b);
// }

//------------------------------------------------------------------------------
// sigmoid Float16
template<typename T>
__device__ inline T sigmoid(const T& a) { return T(1) / (T(1) + exp(-a)); }

__device__ inline __half2 sigmoid(const __half2& a) {
    const __half2 one = __half2(1,1);
    return one / (one + h2exp(-a));
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

#endif // mathSupplemental_h