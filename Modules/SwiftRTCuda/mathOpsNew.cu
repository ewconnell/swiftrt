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
#include "disp.h"
#include "mathOps.h"
#include "mathSupplemental.h"
#include <type_traits>

//------------------------------------------------------------------------------
// greaterElements
template<typename T>
__device__ inline bool greaterElements(const T& a, const T& b) {
    return a > b;
}


//==============================================================================
// Swift importable C interface functions
//==============================================================================

OpT(Abs, abs, (isNumeric<T>() && isSame<T,Out>()) || isComplex<T>())

cudaError_t srtAbs(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Abs>(a, aDesc, out, oDesc, stream);
}

OpT(Acos, acos, (isFloating<T>() && isSame<T,Out>()))

// Must be promoted types
cudaError_t srtAcos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Acos>(a, aDesc, out, oDesc, stream);
}

OpTT(Add, add, (isSame<T,Out>() && isNumeric<T>()))

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Add>(a, aDesc, b, bDesc, out, oDesc, stream);
}

OpTT(Greater, greaterElements, (isNumeric<T>() && isBool<Out>()))

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Greater>(a, aDesc, b, bDesc, out, oDesc, stream);
}
