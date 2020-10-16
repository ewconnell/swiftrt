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
#include <cuda_runtime.h>
#include <cublasLt.h>

/* Set up function decorations */
#ifndef __DEVICE_INLINE__
#if defined(__CUDACC__)
#define __DEVICE_INLINE__ __device__ __forceinline__
#define __HOSTDEVICE_INLINE__ __host__ __device__ __forceinline__
#else /* !defined(__CUDACC__) */
#define __DEVICE_INLINE__
#define __HOSTDEVICE_INLINE__
#endif /* defined(__CUDACC_) */
#endif

// make visible to Swift as C API
#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
//  srtDataTypes
typedef enum {
    unknown     = -1,
    // floating point types
    real16F     = CUDA_R_16F,
    real16BF    = CUDA_R_16BF,
    real32F     = CUDA_R_32F,
    real64F     = CUDA_R_64F,
    complex16F  = CUDA_C_16F,
    complex16BF = CUDA_C_16BF,
    complex32F  = CUDA_C_32F,
    complex64F  = CUDA_C_64F,

    // integral types
    real1U,
    real4I      = CUDA_R_4I,
    real4U      = CUDA_R_4U,
    real8I      = CUDA_R_8I,
    real8U      = CUDA_R_8U, 
    real16I     = CUDA_R_16I,
    real16U     = CUDA_R_16U,
    real32I     = CUDA_R_32I,
    real32U     = CUDA_R_32U, 
    real64U     = CUDA_R_64U,
    real64I     = CUDA_R_64I,
    complex4I   = CUDA_C_4I, 
    complex4U   = CUDA_C_4U,
    complex8I   = CUDA_C_8I, 
    complex8U   = CUDA_C_8U,
    complex16I  = CUDA_C_16I,
    complex16U  = CUDA_C_16U,
    complex32I  = CUDA_C_32I,
    complex32U  = CUDA_C_32U,
    complex64I  = CUDA_C_64I,
    complex64U  = CUDA_C_64U,

    // bool types
    boolean
} srtDataType;

//==============================================================================
#ifdef __cplusplus
}
#endif
