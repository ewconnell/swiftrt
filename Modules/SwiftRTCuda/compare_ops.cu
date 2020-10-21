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
#include "compare_fn.cuh"
#include "compare_vjp.cuh"
#include "op2.cuh"
#include "op3.cuh"
#include "srt_traits.cuh"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
Op2(And, andElements, (isBool<A>() && isBool<Out>()))

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
Op3SwapBC(AlmostEqual, almostEqual, (isBool<Out>() && (isFloating<A>() || isComplex<A>())))

cudaError_t srtElementsAlmostEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* tolerance,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<AlmostEqual>(a, aDesc, tolerance, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Equal, equal, (isEquatable<A>() && isBool<Out>()))

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc);
    return select<Equal>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Greater, greater, (isComparable<A>() && isBool<Out>()))

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectTT_Bool<Greater>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtGreaterTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectTT_Bool<Greater>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(GreaterOrEqual, greaterOrEqual, (isComparable<A>() && isBool<Out>()))

cudaError_t srtGreaterOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectTT_Bool<GreaterOrEqual>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtGreaterOrEqualTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectTT_Bool<GreaterOrEqual>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(Less, less, (isComparable<A>() && isBool<Out>()))

cudaError_t srtLess(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectTT_Bool<Less>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtLessTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectTT_Bool<Less>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(LessOrEqual, lessOrEqual, (isComparable<A>() && isBool<Out>()))

cudaError_t srtLessOrEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectTT_Bool<LessOrEqual>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtLessOrEqualTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectTT_Bool<LessOrEqual>(a, aDesc, element, out, oDesc, stream);
}
    
//------------------------------------------------------------------------------
Op2(MinElements, minElements, (isComparable<A>() && isSame<A, Out>()))

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
Op2(MaxElements, maxElements, (isComparable<A>() && isSame<A, Out>()))

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
Op2(NotEqual, notEqual, (isEquatable<A>() && isBool<Out>()))

cudaError_t srtNotEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<NotEqual>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Or, orElements, (isBool<A>() && isBool<Out>()))

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
Op3(Replace, conditionalAssign, isBool<C>())

cudaError_t srtReplace(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* condition, const srtTensorDescriptor* pcDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABC(paDesc, pbDesc, pcDesc, poDesc)
    return selectTTBool_T<Replace>(b, bDesc, a, aDesc, condition, cDesc, out, oDesc, stream);
}

//==============================================================================
Op3(VjpMin, vjpMin, (isComparable<A>() && isSame<A,Out>()))

cudaError_t srtVjpMin(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABC(paDesc, pbDesc, pcDesc, poDesc)
    return select<VjpMin>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
}

cudaError_t srtVjpMinTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* e,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<VjpMin>(a, aDesc, e, b, bDesc, out, oDesc, stream);
}

Op32(VjpMin2, vjpMin, (isComparable<A>() && isSame<A,Out>()))

cudaError_t srtVjpMinOO(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* outT, const srtTensorDescriptor* po0Desc,
    void* outF, const srtTensorDescriptor* po1Desc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABCOO(paDesc, pbDesc, pcDesc, po0Desc, po1Desc)
    return select<VjpMin2>(a, aDesc, b, bDesc, c, cDesc, outT, o0Desc, outF, o1Desc, stream);
}

cudaError_t srtVjpMinTEOO(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* outT, const srtTensorDescriptor* po0Desc,
    void* outF, const srtTensorDescriptor* po1Desc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAECOO(paDesc, pcDesc, po0Desc, po1Desc)
    return select<VjpMin2>(a, aDesc, b, c, cDesc, outT, o0Desc, outF, o1Desc, stream);
}

//==============================================================================
Op3(VjpMax, vjpMax, (isComparable<A>() && isSame<A,Out>()))

cudaError_t srtVjpMax(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABC(paDesc, pbDesc, pcDesc, poDesc)
    return select<VjpMax>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
}

cudaError_t srtVjpMaxTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* e,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<VjpMax>(a, aDesc, e, b, bDesc, out, oDesc, stream);
}

Op32(VjpMax2, vjpMax, (isComparable<A>() && isSame<A,Out>()))

cudaError_t srtVjpMaxOO(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* outT, const srtTensorDescriptor* po0Desc,
    void* outF, const srtTensorDescriptor* po1Desc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABCOO(paDesc, pbDesc, pcDesc, po0Desc, po1Desc)
    return select<VjpMax2>(a, aDesc, b, bDesc, c, cDesc, outT, o0Desc, outF, o1Desc, stream);
}

cudaError_t srtVjpMaxTEOO(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* outT, const srtTensorDescriptor* po0Desc,
    void* outF, const srtTensorDescriptor* po1Desc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAECOO(paDesc, pcDesc, po0Desc, po1Desc)
    return select<VjpMax2>(a, aDesc, b, c, cDesc, outT, o0Desc, outF, o1Desc, stream);
}
