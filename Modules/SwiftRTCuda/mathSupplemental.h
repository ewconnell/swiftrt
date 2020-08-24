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
// supplemental functions
//==============================================================================

//------------------------------------------------------------------------------
// abs
__device__ inline __half abs(const __half& a) { return __habs(a); }
__device__ inline __half2 abs(const __half2& a) { return __habs2(a); }

__device__ inline __nv_bfloat16 abs(const __nv_bfloat16& a) {
    #if (__CUDA_ARCH__ < 800)
    return abs(float(a));
    #else
    return __habs(a);
    #endif
}

__device__ inline __nv_bfloat162 abs(const __nv_bfloat162& a) {
    #if (__CUDA_ARCH__ < 800)
    return __nv_bfloat162(abs(float(a.x)), abs(float(a.y)));
    #else
    return __habs2(a);
    #endif
}

//------------------------------------------------------------------------------
// acos
__device__ inline __half acos(const __half& a) { return acosf(a); }
__device__ inline __half2 acos(const __half2& a) { 
    return __half2(acosf(a.x), acosf(a.y));
}

__device__ inline __nv_bfloat16 acos(const __nv_bfloat16& a) {
    return acosf(a);
}

__device__ inline __nv_bfloat162 acos(const __nv_bfloat162& a) {
    return __nv_bfloat162(acosf(a.x), acosf(a.y));
}

//------------------------------------------------------------------------------
// acosh
__device__ inline __half acosh(const __half& a) { return acoshf(a); }
__device__ inline __half2 acosh(const __half2& a) { 
    return __half2(acoshf(a.x), acoshf(a.y));
}

__device__ inline __nv_bfloat16 acosh(const __nv_bfloat16& a) {
    return acoshf(a);
}

__device__ inline __nv_bfloat162 acosh(const __nv_bfloat162& a) {
    return __nv_bfloat162(acoshf(a.x), acoshf(a.y));
}

//------------------------------------------------------------------------------
// exp
__device__ inline __half exp(const __half& a) { return hexp(a); }
__device__ inline __half2 exp(const __half2& a) { return h2exp(a); }

__device__ inline __nv_bfloat16 exp(const __nv_bfloat16& a) {
    #if (__CUDA_ARCH__ < 800)
    return expf(a);
    #else
    return hexp(a);
    #endif
}

__device__ inline __nv_bfloat162 exp(const __nv_bfloat162& a) {
    #if (__CUDA_ARCH__ < 800)
    return __nv_bfloat162(expf(a.x), expf(a.y));
    #else
    return h2exp(a);
    #endif
}


#endif // mathSupplemental_h