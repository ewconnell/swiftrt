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

#include "compareOps.h"
#include "dispatchHelpers.h"

//------------------------------------------------------------------------------
// andElements 
__device__ inline uchar4 andElements(const uchar4& a, const uchar4& b) {
    const uint32_t out = UINT32_CREF(a) & UINT32_CREF(b);
    return CAST(uchar4, out);
}

// orElements 
__device__ inline uchar4 orElements(const uchar4& a, const uchar4& b) {
    const uint32_t out = UINT32_CREF(a) | UINT32_CREF(b);
    return CAST(uchar4, out);
}

//==============================================================================
// ops
//==============================================================================

#define BOOLEANOP(OpName, name) \
template<typename T> struct OpName: OpBase<T,T> { \
    __device__ inline static T op(const T& a, const T& b) { return name(a, b); } \
}; \

#define COMPAREOP(OpName, name) \
template<typename T> struct OpName: OpBase<T,bool> { \
    __device__ inline static bool op(const T& a, const T& b) { return name(a, b); } \
}; \

BOOLEANOP(And, andElements)
BOOLEANOP(Or, orElements)

COMPAREOP(Equal, equalElements)
COMPAREOP(Greater, greaterElements)
COMPAREOP(GreaterOrEqual, greaterOrEqualElements)
COMPAREOP(Less, lessElements)
COMPAREOP(LessOrEqual, lessOrEqualElements)
COMPAREOP(Max, maxElements)
COMPAREOP(Min, minElements)
COMPAREOP(NotEqual, notEqualElements)


//==============================================================================
// kernels
//==============================================================================

//==============================================================================
// dynamic dispatch functions
//==============================================================================

//==============================================================================
// Swift importable C interface functions
//==============================================================================

cudaError_t srtAnd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectLogical<And>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtElementsAlmostEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* tolerance,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLessOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtMax(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtMin(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtNotEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtOr(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectLogical<Or>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtReplace(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* condition, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}
