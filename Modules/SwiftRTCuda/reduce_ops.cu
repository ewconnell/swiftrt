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
#include "index.h"
#include "reduce_fn.h"
#include "op1.h"
#include <cstddef>


// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtReduceAll(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    const size_t* axes,
    const size_t  axesCount,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

//==============================================================================

template<typename T>
inline cudaError_t sum(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    const size_t* axes,
    const size_t  axesCount,
    cudaStream_t stream
) {
    const T* a = static_cast<const T*>(pA);
    T* out = static_cast<T*>(pOut);
    int count = aDesc.count;

    if (axes == NULL) {
        void   *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a, out, count));
        CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
        // Run
        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, a, out, count));
        if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    }
    return cudaSuccess;
}

cudaError_t srtReduceSum(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    const size_t* axes,
    const size_t  axesCount,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    if (!(aDesc.isDense() && oDesc.isDense())) return cudaErrorNotSupported;
    switch(aDesc.type) {
        case real32F:  return sum<float>(a, aDesc, out, oDesc, axes, axesCount, stream);
        // case real16F:  return selectOut<Op, float16>(a, aDesc, out, oDesc, stream);
        // case real16BF: return selectOut<Op, bfloat16>(a, aDesc, out, oDesc, stream);
        // case real64F:  return selectOut<Op, double>(a, aDesc, out, oDesc, stream);
        // case real32I:  return selectOut<Op, int32_t>(a, aDesc, out, oDesc, stream);
        // case real8U:   return selectOut<Op, uint8_t>(a, aDesc, out, oDesc, stream);
        // case real8I:   return selectOut<Op, int8_t>(a, aDesc, out, oDesc, stream);
        // case real16U:  return selectOut<Op, uint16_t>(a, aDesc, out, oDesc, stream);
        // case real16I:  return selectOut<Op, int16_t>(a, aDesc, out, oDesc, stream);
        // case boolean:  return selectOut<Op, bool>(a, aDesc, out, oDesc, stream);
        // case complex32F: return selectOut<Op, complexf>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
