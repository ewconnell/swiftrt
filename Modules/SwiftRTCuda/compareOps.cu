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
#include "compareOps.h"
#include "compareSupplemental.h"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
Op2(And, andElements, (isBool<T>() && isBool<Out>()))

cudaError_t srtAnd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<And>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op3(AlmostEqual, almostEqual, (isNumeric<T>() && isBool<Out>()))

cudaError_t srtElementsAlmostEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* tolerance,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<AlmostEqual>(a, aDesc, b, bDesc, tolerance, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Equal, equalElements, (isEquatable<T>() && isBool<Out>()))

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Equal>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Greater, greaterElements, (isComparable<T>() && isBool<Out>()))

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Greater>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtGreaterTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Greater>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(GreaterOrEqual, greaterOrEqualElements, (isComparable<T>() && isBool<Out>()))

cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<GreaterOrEqual>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtGreaterOrEqualTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<GreaterOrEqual>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(Less, lessElements, (isComparable<T>() && isBool<Out>()))

cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Less>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtLessTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Less>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(LessOrEqual, lessOrEqualElements, (isComparable<T>() && isBool<Out>()))

cudaError_t srtLessOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<LessOrEqual>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtLessOrEqualTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<LessOrEqual>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(MinElements, minElements, (isComparable<T>() && isSame<T, Out>()))

cudaError_t srtMin(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<MinElements>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtMinTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<MinElements>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(MaxElements, maxElements, (isComparable<T>() && isSame<T, Out>()))

cudaError_t srtMax(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<MaxElements>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtMaxTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<MaxElements>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(NotEqualElements, notEqualElements, (isEquatable<T>() && isBool<Out>()))

cudaError_t srtNotEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<NotEqualElements>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Or, orElements, (isBool<T>() && isBool<Out>()))

cudaError_t srtOr(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Or>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
OpTTU(Replace, conditionalAssign, (isSame<T,Out>() && isBool<U>()))

cudaError_t srtReplace(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* condition, const srtTensorDescriptor* pcDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABC(paDesc, pbDesc, pcDesc, poDesc)
    return select<Replace>(b, bDesc, a, aDesc, condition, cDesc, out, oDesc, stream);
}
