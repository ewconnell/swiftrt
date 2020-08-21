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
#ifndef mathOpFunctions_h
#define mathOpFunctions_h

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

//==============================================================================
// math function additions for consistency
//==============================================================================

//==============================================================================
// abs(x)
__device__ inline static __half abs(const __half& x) {
    return __half(abs(float(x)));
}

__device__ inline static __half2 abs(const __half2& x) { return __habs2(x); }

//--------------------------------------
// BFloat16
// __device__ inline static __nv_bfloat162 abs(const __nv_bfloat162& x) {
//     return __float2bfloat16_rn(abs(__bfloat162float(x)));
// }

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
__device__ inline static __nv_bfloat162 abs(const __nv_bfloat162& x) {
    return __habs2(x);
}
#else
__device__ inline __nv_bfloat162 abs(const __nv_bfloat162& x) {
    __nv_bfloat162 c;
    c.x = __float2bfloat16_rn(abs(__bfloat162float(x.x)));
    c.y = __float2bfloat16_rn(abs(__bfloat162float(x.y)));
    return c;
}
#endif



#endif // mathOpFunctions_h