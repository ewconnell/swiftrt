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
#define __CUDA_DEVICE__ __device__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* !defined(__CUDACC__) */
#define __CUDA_HOSTDEVICE__
#define __CUDA_DEVICE__
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

__CUDA_DEVICE__ inline float162 init_float162(const float16& x, const float16& y) {
    float162 v; v.x = x; v.y = y; return v;
}

typedef __nv_bfloat16  bfloat16;
typedef __nv_bfloat162 bfloat162;

__CUDA_DEVICE__ inline bfloat162 init_bfloat162(const bfloat16& x, const bfloat16& y) {
    bfloat162 v; v.x = x; v.y = y; return v;
}

//==============================================================================
// supplemental logical types
struct bool2 {
    bool x, y;
    __CUDA_HOSTDEVICE__ inline bool2(bool _x, bool _y) { x = _x; y = _y; }
    
    __CUDA_DEVICE__ inline bool2(float162 v) { x = v.x; y = v.y; }
    __CUDA_DEVICE__ inline bool2(bfloat162 v) { x = v.x; y = v.y; }
    __CUDA_DEVICE__ inline bool2(unsigned v) {
        x = v & 0xFF;
        y = (v >> 16) & 0xFF;
    }
};

struct bool4 {
    bool x, y, z, w;
    __CUDA_HOSTDEVICE__ inline bool4(bool _x, bool _y, bool _z, bool _w) {
        x = _x; y = _y; z = _z; w = _w;
    }
    __CUDA_HOSTDEVICE__ inline bool4(unsigned v) {
        *this = *reinterpret_cast<const bool4*>(&v);
    }
};
