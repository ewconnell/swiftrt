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

#include <driver_types.h>

//==============================================================================
// kernel helpers
#define GRID_STRIDE_LOOP(i, n)                                                 \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n);          \
       i += blockDim.x * gridDim.x)

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

#endif // __kernelHelpers_h__