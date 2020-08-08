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
inline bool isDense(const srtTensorDescriptor& t) { return t.count == t.spanCount; }

//==============================================================================
// Index
template<size_t Rank>
struct Index {
    const uint32_t shape[Rank];
    const uint32_t strides[Rank];

    // initializer
    __host__ Index(const srtTensorDescriptor& tensor) {
        for (int i = 0; i < Rank; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            shape[i] = uint32_t(tensor.shape[i]);
            strides[i] = uint32_t(tensor.strides[i]);
        }
    }

    __device__ uint32_t linear(dim3 pos) {
        return 0;
    }
};

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

#define GRID_LOOP_STRIDED(ai, sa, bi, sb, oi, n) \
    int ti = blockIdx.x * blockDim.x + threadIdx.x; \
    int step = blockDim.x * gridDim.x; \
    int aStep = step * (sa); \
    int bStep = step * (sb); \
    for(int ai = ti * (sa), bi = ti * (sb), oi = ti; \
        oi < (n); ai += aStep, bi += bStep, oi += step)

// threads per block
const unsigned THREADS_PER_BLOCK = 1024;

// number of blocks for threads.
inline unsigned BLOCK_COUNT(unsigned N) {
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

inline unsigned shiftDownRoundingUp(unsigned num, unsigned shift) 
{
    unsigned count = (num + (1 << shift) - 1) >> shift;
    return count;
}

#endif // __kernelHelpers_h__