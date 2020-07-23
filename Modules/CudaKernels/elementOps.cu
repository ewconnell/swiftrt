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
#include "include/elementOps.h"
#include "include/kernelHelpers.h"

//------------------------------------------------------------------------------
// device kernel
template<typename T>
__global__ void srtAdd_kernel(
    const void *va,
    const void *vb,
    void *vc,
    unsigned count
) {
    const T* a = (T*)va;
    const T* b = (T*)vb;
    T* c = (T*)vc;
    KERNEL_LOOP(i, count) {
        c[i] = a[i] + b[i];
    }
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t srtAdd(
    cudaDataType_t type,
    const void *a,
    const void *b,
    void *c,
    size_t count,
    cudaStream_t stream
) {
    KernelPreCheck(stream);
    unsigned blocks = BLOCK_COUNT(count);
    unsigned threads = THREADS_PER_BLOCK;
    switch(type) {
        case CUDA_R_8I: srtAdd_kernel<char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_8U: srtAdd_kernel<unsigned char> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16I: srtAdd_kernel<short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16U: srtAdd_kernel<unsigned short> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16F: srtAdd_kernel<__half> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_16BF: srtAdd_kernel<__nv_bfloat16> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_32F: srtAdd_kernel<float> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        case CUDA_R_64F: srtAdd_kernel<double> <<<blocks, threads, 0, stream>>>(a, b, c, count); break;
        default: printf("cudaDataType_t not implemented"); assert(false);
    }
    return KernelPostCheck(stream);
}
