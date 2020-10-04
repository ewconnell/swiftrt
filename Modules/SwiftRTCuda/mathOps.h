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
#ifndef mathOps_h
#define mathOps_h

#include "commonCDefs.h"

// make visible to Swift as C API
#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
//
cudaError_t srtAbs(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAcos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAddTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAtan2(
    const void* y, const srtTensorDescriptor* yDesc,
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtDivTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtDivET(
    const void* element,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtHypot(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLog(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMulTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMultiplyAdd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* c, const srtTensorDescriptor* cDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtMultiplyAddTTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtPow(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtPowTE(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* exponent,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSign(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSqrt(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSquared(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSubTE(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* element,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtSubET(
    const void* element,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtTan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

cudaError_t srtTanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream);

//==============================================================================
#ifdef __cplusplus
}
#endif

#endif // mathOps_h