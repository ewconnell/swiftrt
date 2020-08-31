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

//==============================================================================
// ops
//==============================================================================

#define COMPAREOP2(OpName, name) \
template<typename T> struct OpName: OpBase<T,bool> { \
    __device__ inline static bool op(const T& a, const T& b) { return name(a, b); } \
}; \

COMPAREOP2(And, andElements)
COMPAREOP2(Equal, equalElements)
COMPAREOP2(Greater, greaterElements)
COMPAREOP2(GreaterOrEqual, greaterOrEqualElements)
COMPAREOP2(Less, lessElements)
COMPAREOP2(LessOrEqual, lessOrEqualElements)
COMPAREOP2(Max, maxElements)
COMPAREOP2(Min, minElements)
COMPAREOP2(NotEqual, notEqualElements)
COMPAREOP2(Or, orElements)


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
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtElementsAlmostEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* tolerance,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtLessOrEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtMax(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtMin(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtNotEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtOr(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}

cudaError_t srtReplace(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* condition, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return cudaErrorNotSupported;
}
