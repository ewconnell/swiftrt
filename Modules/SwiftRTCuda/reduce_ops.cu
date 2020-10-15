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
#include "precomp.hpp"

#include "index.h"
#include "reduce_fn.h"
#include "op1.h"
#include "srt_cdefs.h"
#include <cstddef>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtAbsSum(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<AbsSumOp>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtAll(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    assert(aDesc.type == boolean && oDesc.type == boolean);
    return reduce<AllOp, bool>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtAny(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    assert(aDesc.type == boolean && oDesc.type == boolean);
    return reduce<AnyOp, bool>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtSum(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<SumOp>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtMean(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtMinValue(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<MinOp>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtArgMin(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<SumOp>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtMaxValue(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectType<MaxOp>(a, aDesc, out, oDesc, g_allocator, stream);
}

cudaError_t srtArgMax(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtProd(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtProdNonZeros(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
