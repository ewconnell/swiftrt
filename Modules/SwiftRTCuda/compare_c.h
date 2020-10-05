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
#include "tensor_descriptor.h"

// make visible to Swift as C API
#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
//
cudaError_t srtAnd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtElementsAlmostEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* tolerance,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtGreaterTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtGreaterOrEqualTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLessTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLessOrEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLessOrEqualTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMin(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMinTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMax(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMaxTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtNotEqual(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtOr(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtReplace(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* condition, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

//==============================================================================

cudaError_t srtVjpMin(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* c, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtVjpMinTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtVjpMinOO(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* c, const srtTensorDescriptor* cDesc,
    void* outT, const srtTensorDescriptor* oTDesc,
    void* outF, const srtTensorDescriptor* oFDesc,
    cudaStream_t stream);

cudaError_t srtVjpMinTEOO(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* cDesc,
    void* outT, const srtTensorDescriptor* oTDesc,
    void* outF, const srtTensorDescriptor* oFDesc,
    cudaStream_t stream);

cudaError_t srtVjpMax(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* c, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtVjpMaxTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtVjpMaxOO(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* c, const srtTensorDescriptor* cDesc,
    void* outT, const srtTensorDescriptor* oTDesc,
    void* outF, const srtTensorDescriptor* oFDesc,
    cudaStream_t stream);

cudaError_t srtVjpMaxTEOO(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* cDesc,
    void* outT, const srtTensorDescriptor* oTDesc,
    void* outF, const srtTensorDescriptor* oFDesc,
    cudaStream_t stream);

//==============================================================================
#ifdef __cplusplus
}
#endif
