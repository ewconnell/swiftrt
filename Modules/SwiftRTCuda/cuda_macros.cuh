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
#include <cuda_runtime.h>

/* Set up function decorations */
#ifndef __DEVICE_INLINE__
#if defined(__CUDACC__)
#define __DEVICE_INLINE__ __device__ __forceinline__
#define __HOSTDEVICE_INLINE__ __host__ __device__ __forceinline__
#else /* !defined(__CUDACC__) */
#define __DEVICE_INLINE__ inline
#define __HOSTDEVICE_INLINE__ inline
#endif /* defined(__CUDACC_) */
#endif

//==============================================================================
// launch error detection
inline void CudaKernelPreCheck(cudaStream_t stream) {
#ifdef DEBUG
	// reset error variable to cudaSuccess
	cudaGetLastError();
#endif
}

inline cudaError_t CudaKernelPostCheck(cudaStream_t stream) {
#ifdef DEBUG
	cudaStreamSynchronize(stream);
	return cudaGetLastError();
#else
	return cudaSuccess;
#endif
}

//==============================================================================
// used for casting between gpu simd types and uint32_t
#define UINT_CREF(_v) reinterpret_cast<const unsigned&>(_v)
#define CAST(type, _v) (*reinterpret_cast<const type*>(&(_v)))

