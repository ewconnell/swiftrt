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
#include "compareSupplemental.h"
#include "dispatchHelpers.h"

//==============================================================================
// ops
//==============================================================================

#define BOOL_OP(OpName, name) \
template<typename T> struct OpName: OpBase<T,T> { \
    __device__ inline static T op(const T& a, const T& b) { return name(a, b); } \
}; \

#define COMPARE_OP(OpName, name) \
template<typename In, typename Out> struct OpName: OpBase<In,Out> { \
    __device__ inline static Out op(const In& a, const In& b) { return name(a, b); } \
}; \

BOOL_OP(And, andElements)
BOOL_OP(Or, orElements)

COMPARE_OP(Equal, equalElements)
COMPARE_OP(Greater, greaterElements)
COMPARE_OP(GreaterOrEqual, greaterOrEqualElements)
COMPARE_OP(Less, lessElements)
COMPARE_OP(LessOrEqual, lessOrEqualElements)
COMPARE_OP(Max, maxElements)
COMPARE_OP(Min, minElements)
COMPARE_OP(NotEqual, notEqualElements)


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
    return selectBool<And>(a, aDesc, b, bDesc, out, oDesc, stream);
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
    cudaStream_t stream
) {
    // Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    // return selectIntFloatingPacked<Equal>(a, aDesc, b, bDesc, out, oDesc, stream);
    return cudaErrorNotSupported;
}

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtGreaterTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
    
cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtGreaterOrEqualTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}
    
cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

cudaError_t srtLessTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
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

cudaError_t srtLessOrEqualTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
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

cudaError_t srtMinTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
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

cudaError_t srtMaxTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
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
    return selectBool<Or>(a, aDesc, b, bDesc, out, oDesc, stream);
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
