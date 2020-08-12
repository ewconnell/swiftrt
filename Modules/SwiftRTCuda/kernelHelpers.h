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
#if !defined(__kernelHelpers_h__)
#define __kernelHelpers_h__

#include <assert.h>
#include "commonCDefs.h"

//==============================================================================
// TensorDescriptor
// C++ enhanced wrapper
struct TensorDescriptor: srtTensorDescriptor {
    inline bool isDense() const { return count == spanCount; }
    inline bool isSingle() const { return spanCount == 1; }
};

static_assert(sizeof(TensorDescriptor) == sizeof(srtTensorDescriptor),
    "TensorDescriptor is a c++ wrapper and cannot contain additional members");

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// threads per block
const unsigned THREADS_PER_BLOCK = 1024;

// number of blocks for threads
inline unsigned BLOCK_COUNT(unsigned n) {
  return ((n) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

//==============================================================================
// 
inline unsigned shiftDownRoundingUp(unsigned num, unsigned shift) 
{
    unsigned count = (num + (1 << shift) - 1) >> shift;
    return count;
}

//==============================================================================
// #if (__CUDA_ARCH__ < 800)
// __device__ __forceinline__ __nv_bfloat162 operator+(const __nv_bfloat162& l, const __nv_bfloat162& r) {
//     __nv_bfloat162 c;
//     c.x = __float2bfloat16_rn(__bfloat162float(l.x) + __bfloat162float(r.x));
//     c.y = __float2bfloat16_rn(__bfloat162float(l.y) + __bfloat162float(r.y));
//     return c;
// }
// #endif

__device__ inline __nv_bfloat162 add(const __nv_bfloat162& l, const __nv_bfloat162& r) {
    __nv_bfloat162 c;
    c.x = __float2bfloat16_rn(__bfloat162float(l.x) + __bfloat162float(r.x));
    c.y = __float2bfloat16_rn(__bfloat162float(l.y) + __bfloat162float(r.y));
    return c;
}

// template<typename E>
// __global__ void add_bfloat162(
//     const void *va, int strideA,
//     const void *vb, int strideB,
//     void *vc,
//     unsigned count
// ) {
//     auto a = static_cast<const E*>(va);
//     auto b = static_cast<const E*>(vb);
//     auto c = static_cast<E*>(vc);

//     GRID_STRIDE_LOOP(ai, strideA, bi, strideB, ci, count) {
//         #if (__CUDA_ARCH__ >= 800)
//             c[ci] = a[ai] + b[bi];
//         #else
//             c[ci] = add(a[ai], b[bi]);
//         #endif
//     }
// }


#endif // __kernelHelpers_h__