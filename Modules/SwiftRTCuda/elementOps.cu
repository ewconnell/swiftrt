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
#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "elementOps.h"
#include "kernelHelpers.h"

//------------------------------------------------------------------------------
// add
template<typename T>
__global__ void add(const void *va, const void *vb, void *vc, unsigned count) {
    const T* a = (T*)va; const T* b = (T*)vb; T* c = (T*)vc;
    GRID_STRIDE_LOOP(i, count) {
        c[i] = a[i] + b[i];
    }
}

// add2_Float16
// TODO: Is there a way to do a __hadd8?? 
__global__ void add2_Float16(const void *va, const void *vb, void *vc, unsigned count) {
    const __half2 *a = (__half2 *)va;
    const __half2 *b = (__half2 *)vb;
    __half2 *c = (__half2 *)vc;

    GRID_STRIDE_LOOP(i, count) {
        c[i] = __hadd2(a[i], b[i]);
    }
}

// add4_Float32
__global__ void add4_Float32(const void *va, const void *vb, void *vc, unsigned count) {
    const float4 *a = (float4 *)va;
    const float4 *b = (float4 *)vb;
    float4 *c = (float4 *)vc;

    GRID_STRIDE_LOOP(i, count) { c[i] = a[i] + b[i]; }
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t srtAdd(cudaDataType_t type, const void *a, const void *b, void *c,
                    unsigned count, cudaStream_t stream) {
    KernelPreCheck(stream);
    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;
    switch(type) {
        case CUDA_R_8I: add<char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_8U: add<unsigned char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16I: add<short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16U: add<unsigned short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16F: {
            int elements = shiftDownRoundingUp(count, 1);
            add2_Float16<<<BLOCK_COUNT(elements), threads, 0, stream>>>(a, b, c, elements); 
            break;
        }

        case CUDA_R_32F: {
            int elements = shiftDownRoundingUp(count, 2);
            add4_Float32<<<BLOCK_COUNT(elements), threads, 0, stream>>>(a, b, c, elements);
            break;
        }

        case CUDA_R_64F: add<double> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        default: printf("cudaDataType_t not implemented"); assert(false);
    }
    return KernelPostCheck(stream);
}
