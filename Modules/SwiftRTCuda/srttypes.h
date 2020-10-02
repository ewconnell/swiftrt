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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/* Set up function decorations */
#ifndef __CUDA_HOSTDEVICE__
#if defined(__CUDACC__)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* !defined(__CUDACC__) */
#define __CUDA_HOSTDEVICE__
#endif /* defined(__CUDACC_) */
#endif

//==============================================================================
// used for casting between gpu simd types and uint32_t
#define UINT_CREF(_v) reinterpret_cast<const unsigned&>(_v)
#define CAST(type, _v) (*reinterpret_cast<const type*>(&(_v)))

//==============================================================================
// half precision real types
typedef __half  float16;
typedef __half2 float162;

typedef __nv_bfloat16  bfloat16;
typedef __nv_bfloat162 bfloat162;

//==============================================================================
// supplemental logical types
struct bool2 {
    bool b0, b1;
    __CUDA_HOSTDEVICE__ inline bool2(bool v0, bool v1) { b0 = v0; b1 = v1; }
    
    __device__ inline bool2(float162 v) { b0 = v.x; b1 = v.y; }
    __device__ inline bool2(bfloat162 v) { b0 = v.x; b1 = v.y; }
    __device__ inline bool2(unsigned v) {
        b0 = v & 0xFF;
        b1 = (v >> 16) & 0xFF;
    }
};

struct bool4 {
    bool b0, b1, b2, b3;
    __CUDA_HOSTDEVICE__ inline bool4(bool v0, bool v1, bool v2, bool v3) {
        b0 = v0; b1 = v1; b2 = v2; b3 = v3;
    }
    __CUDA_HOSTDEVICE__ inline bool4(unsigned v) {
        *this = *reinterpret_cast<const bool4*>(&v);
    }
};
