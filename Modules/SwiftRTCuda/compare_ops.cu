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
#include "op2.h"
#include "op3.h"
#include "srt_traits.cuh"
#include "tensor_api.h"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
LogicalOp2(And, andElements, isBool<A>())

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
LogicalOp2(Equal, equal, isEquatable<A>())

cudaError_t srtEqual(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc);
    return select<Equal>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtEqualFlat(
    srtDataType type,
    const void* a,
    const void* b,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<Equal>(type, a, b, boolean, out, count, stream);
}

cudaError_t srtEqualTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc);
    return select<Equal>(a, aDesc, element, out, oDesc, stream);
}

cudaError_t srtEqualFlatTE(
    srtDataType type,
    const void* a,
    const void* element,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return selectTE<Equal>(type, a, element, boolean, out, count, stream);
}

//------------------------------------------------------------------------------
LogicalOp2(Greater, greater, isComparable<A>())

cudaError_t srtGreater(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Greater>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtGreaterFlat(
    srtDataType type,
    const void* a,
    const void* b,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<Greater>(type, a, b, boolean, out, count, stream);
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
    
cudaError_t srtGreaterFlatTE(
    srtDataType type,
    const void* a,
    const void* element,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return selectTE<Greater>(type, a, element, boolean, out, count, stream);
}

//------------------------------------------------------------------------------
LogicalOp2(GreaterOrEqual, greaterOrEqual, isComparable<A>())

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
LogicalOp2(Less, less, isComparable<A>())

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
LogicalOp2(LessOrEqual, lessOrEqual, isComparable<A>())

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
Op2(MinElements, minElements, isComparable<A>())

cudaError_t srtMin(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<MinElements>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtMinFlat(
    srtDataType type,
    const void* a,
    const void* b,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<MinElements>(type, a, b, type, out, count, stream);
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

cudaError_t srtMinFlatTE(
    srtDataType type,
    const void* a,
    const void* element,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return selectTE<MinElements>(type, a, element, type, out, count, stream);
}

//------------------------------------------------------------------------------
Op2(MaxElements, maxElements, isComparable<A>())

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
LogicalOp2(NotEqual, notEqual, isEquatable<A>())

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
LogicalOp2(Or, orElements, isBool<A>())

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
    // a and b are reversed because conditionalAssign has opposite behavior of replace
    return selectTTBool_T<Replace>(b, bDesc, a, aDesc, condition, cDesc, out, oDesc, stream);
}

cudaError_t srtReplaceFlat(
    srtDataType type,
    const void* a,
    const void* b,
    srtDataType ctype,
    const void* condition,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    // a and b are reversed because conditionalAssign has opposite behavior of replace
    return select<Replace>(type, b, a, ctype, condition, out, count, stream);
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
