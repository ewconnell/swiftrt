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
#include <driver_types.h>
#include <vector_functions.h>

//==============================================================================
// kernel helpers
// #define GRID_STRIDE_LOOP1(i, n)                                                \
//   for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n);          \
//        i += blockDim.x * gridDim.x)

#define GRID_STRIDE_LOOP(ai, sa, bi, sb, ci, n) \
    int ti = blockIdx.x * blockDim.x + threadIdx.x; \
    int step = blockDim.x * gridDim.x; \
    int aStep = step * (sa); \
    int bStep = step * (sb); \
    for(int ai = ti * (sa), bi = ti * (sb), ci = ti; \
        ci < (n); ai += aStep, bi += bStep, ci += step)

// threads per block
const int THREADS_PER_BLOCK = 1024;

// number of blocks for threads.
inline int BLOCK_COUNT(int N) {
  return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

//==============================================================================
// launch error detection
inline void KernelPreCheck(cudaStream_t stream) {
#ifdef DEBUG
    // reset error variable to cudaSuccess
	cudaGetLastError();
#endif
}

inline cudaError_t KernelPostCheck(cudaStream_t stream)
{
#ifdef DEBUG
    cudaStreamSynchronize(stream);
	return cudaGetLastError();
#else
    return cudaSuccess;
#endif
}

inline int shiftDownRoundingUp(int num, int shift) 
{
    int count = (num + (1 << shift) - 1) >> shift;
    return count;
}

#endif // __kernelHelpers_h__