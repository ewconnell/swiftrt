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
#include <assert.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Element wise Add, Subtract, Multiply, Divide ops
#include "asmdOps.h"

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

// template<typename T>
// __global__ void add_bfloat162(
//     const void *va, int strideA,
//     const void *vb, int strideB,
//     void *vc,
//     unsigned count
// ) {
//     auto a = static_cast<const T*>(va);
//     auto b = static_cast<const T*>(vb);
//     auto c = static_cast<T*>(vc);

//     GRID_STRIDE_LOOP(ai, strideA, bi, strideB, ci, count) {
//         #if (__CUDA_ARCH__ >= 800)
//             c[ci] = a[ai] + b[bi];
//         #else
//             c[ci] = add(a[ai], b[bi]);
//         #endif
//     }
// }

// template<typename T>
// __global__ void addStrided(
//     const void *va, int strideA,
//     const void *vb, int strideB,
//     void *vc, int count
// ) {
//     CPOINTER(a, va); CPOINTER(b, vb); POINTER(c, vc);

//     GRID_STRIDED_LOOP(ai, strideA, bi, strideB, ci, count) {
//         c[ci] = a[ai] + b[bi];
//     }
// }

//==============================================================================
// add
template<typename T>
__global__ void addScalar(const void *va, const void *pScalar, void *vc, int count) {
    CPOINTER(a, va); POINTER(c, vc);
    const T scalar = static_cast<const T*>(pScalar)[0];

    GRID_LOOP(i, count) {
        c[i] = a[i] + scalar;
    }
}

template<typename T>
__global__ void addElements(const void *va, const void *vb, void *vc, int count) {
    CPOINTER(a, va); CPOINTER(b, vb); POINTER(c, vc);

    GRID_LOOP(i, count) {
        c[i] = a[i] + b[i];
    }
}

//------------------------------------------------------------------------------
// add delegates
void srtAddScalar(
    cudaDataType_t type, 
    const void *a, 
    const void *pScalar, 
    void *c,
    int count,
    cudaStream_t stream
) {
    int blocks = BLOCK_COUNT(count);
    int threads = THREADS_PER_BLOCK;

    switch(type) {
        case CUDA_R_32F: addScalar<float><<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        // case CUDA_R_16BF: {
        //     int n = shiftDownRoundingUp(countC, 1);
        //     addScalar_bfloat162<__nv_bfloat162><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, pScalar, c, n);
        //     break;
        // }
        case CUDA_R_16F: {
            int n = shiftDownRoundingUp(count, 1);
            addScalar<__half2><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, pScalar, c, n);
            break;
        }
        case CUDA_R_8I: addScalar<int8_t> <<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        case CUDA_R_8U: addScalar<uint8_t> <<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        case CUDA_R_16I: addScalar<int16_t> <<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        case CUDA_R_16U: addScalar<uint16_t> <<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        case CUDA_R_64F: addScalar<double><<<blocks, threads, 0, stream>>>(a, pScalar, c, count); break;
        default: printf("cudaDataType_t not implemented"); exit(1);
    }
}

void srtAddElements(
    cudaDataType_t type, 
    const void *a, 
    const void *b, 
    void *c, 
    int count,
    cudaStream_t stream
) {
    int blocks = BLOCK_COUNT(count);
    int threads = THREADS_PER_BLOCK;

    switch(type) {
        case CUDA_R_32F: addElements<float><<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        // case CUDA_R_16BF: {
        //     int n = shiftDownRoundingUp(countC, 1);
        //     add_bfloat162<__nv_bfloat162><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, b, c, n);
        //     break;
        // }
        case CUDA_R_16F: {
            int n = shiftDownRoundingUp(count, 1);
            addElements<__half2><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, b, c, n);
            break;
        }
        case CUDA_R_8I: addElements<int8_t> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_8U: addElements<uint8_t> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16I: addElements<int16_t> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16U: addElements<uint16_t> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_64F: addElements<double><<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        default: printf("cudaDataType_t not implemented"); exit(1);
    }
}

//------------------------------------------------------------------------------
// srtAdd
// this function is for dense tensors that can be flattened where
// `isBufferIterable == true`, so strides must equal 0 or 1
cudaError_t srtAdd(
    cudaDataType_t type, 
    const void *a, long strideA, 
    const void *b, long strideB,
    void *c, long count,
    cudaStream_t stream
) {
    // make sure sizes fit within Cuda limitations
    assert(count > 0 && count <= INT32_MAX);
    assert(strideA == 0 || strideA == 1 && strideB == 0 || strideB == 1);
    KernelPreCheck(stream);

    if (strideA == 1 && strideB == 1) {
        srtAddElements(type, a, b, c, count, stream);

    } else if (strideA == 1 && strideB == 0) {
        srtAddScalar(type, a, b, c, count, stream);

    } else if (strideA == 0 && strideB == 1) {
        srtAddScalar(type, b, a, c, count, stream);
    }
    return KernelPostCheck(stream);
}

//------------------------------------------------------------------------------
// srtAddStrided
// performs the operation with fully strided index calculations
cudaError_t srtAddStrided(
    cudaDataType_t type,
    long dims,
    const void *a,
    const int* stridesA, 
    const void *b, 
    const int* stridesB, 
    void *c,
    const int* stridesC, 
    cudaStream_t stream
) {
    // int blocks = BLOCK_COUNT(count);
    // int threads = THREADS_PER_BLOCK;

    // switch(type) {
    //     case CUDA_R_32F: addStrided<float><<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     // case CUDA_R_16BF: {
    //     //     int n = shiftDownRoundingUp(countC, 1);
    //     //     addStrided_bfloat162<__nv_bfloat162><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, b, c, n);
    //     //     break;
    //     // }
    //     case CUDA_R_16F: {
    //         int n = shiftDownRoundingUp(count, 1);
    //         addStrided<__half2><<<BLOCK_COUNT(n), threads, 0, stream>>>(a, b, c, n);
    //         break;
    //     }
    //     case CUDA_R_8I: addStrided<char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     case CUDA_R_8U: addStrided<unsigned char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     case CUDA_R_16I: addStrided<short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     case CUDA_R_16U: addStrided<unsigned short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     case CUDA_R_64F: addStrided<double><<<blocks, threads, 0, stream>>>(a, b, c, count); break;
    //     default: printf("cudaDataType_t not implemented"); exit(1);
    // }

    return cudaSuccess;
}
