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
#include "math_fn.cuh"
#include "op1.h"
#include "op2.h"
#include "op3.h"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
Op1(Abs, abs, ((isSignedNumeric<A>() && isSame<A,Out>()) || isComplexRealType<A,Out>()))

cudaError_t srtAbsFlat(
    srtDataType atype,
    const void* a,
    srtDataType otype,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<Abs>(atype, a, otype, out, count, stream);
}

cudaError_t srtAbs(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    if (aDesc.type == oDesc.type) {
        return select<Abs>(a, aDesc, out, oDesc, stream);
    } else {
        return selectT_O<Abs>(a, aDesc, out, oDesc, stream);
    }
}

//------------------------------------------------------------------------------
Op1(Abs2, abs2, (isComplexRealType<A,Out>()))

cudaError_t srtAbs2Flat(
    srtDataType atype,
    const void* a,
    srtDataType otype,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<Abs2>(atype, a, otype, out, count, stream);
}

cudaError_t srtAbs2(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    if (aDesc.type == oDesc.type) {
        return select<Abs2>(a, aDesc, out, oDesc, stream);
    } else {
        return selectT_O<Abs2>(a, aDesc, out, oDesc, stream);
    }
}

//------------------------------------------------------------------------------
Op1(Acos, acos, isFloating<A>())

cudaError_t srtAcosFlat(
    const void* a, srtDataType atype,
    void* out,
    size_t count, cudaStream_t stream
) {
    return select<Acos>(atype, a, out, count, stream);
}

cudaError_t srtAcos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Acos>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Acosh, acosh, isFloating<A>())

cudaError_t srtAcoshFlat(
    const void* a, srtDataType atype,
    void* out,
    size_t count, cudaStream_t stream
) {
    return select<Acosh>(atype, a, out, count, stream);
}

cudaError_t srtAcosh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Acosh>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Add, add, isNumeric<A>())

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Add>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtAddFlat(
    srtDataType type,
    const void* a,
    const void* b,
    void* out,
    size_t count, 
    cudaStream_t stream
) {
    return select<Add>(type, a, b, type, out, count, stream);
}

//------------------------------------------------------------------------------
cudaError_t srtAddTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Add>(a, aDesc, element, out, oDesc, stream);
}

cudaError_t srtAddTEFlat(
    const void* a, srtDataType atype,
    const void* b,
    void* out,
    size_t count, cudaStream_t stream
) {
    return cudaErrorNotSupported;
}

//------------------------------------------------------------------------------
Op1(Asin, asin, isFloating<A>())

cudaError_t srtAsin(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Asin>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Asinh, asinh, isFloating<A>())

cudaError_t srtAsinh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Asinh>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Atan, atan, isFloating<A>())

cudaError_t srtAtan(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Atan>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Atan2, atan2, isFloating<A>())

cudaError_t srtAtan2(
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    // b comes first
    return select<Atan2>(b, bDesc, a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Atanh, atanh, isFloating<A>())

cudaError_t srtAtanh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Atanh>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Cos, cos, isFloating<A>())

cudaError_t srtCos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Cos>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Cosh, cosh, isFloating<A>())

cudaError_t srtCosh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Cosh>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Div, divide, isNumeric<A>())

cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Div>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtDivTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Div>(a, aDesc, element, out, oDesc, stream);
}

// `true` swaps `a` and `element` when calling `divide`
Op2SwapAB(DivET, divide, (isNumeric<A>() && isSame<A,Out>()))

cudaError_t srtDivET(
    const void* element,
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<DivET>(a, aDesc, element, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Erf, erf, isFloating<A>())

cudaError_t srtErf(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Erf>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Erfc, erfc, isFloating<A>())

cudaError_t srtErfc(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Erfc>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Exp, exp, isFloating<A>())

cudaError_t srtExp(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Exp>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Exp2, exp2, isFloating<A>())

cudaError_t srtExp2(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Exp2>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Exp10, exp10, isFloating<A>())

cudaError_t srtExp10(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Exp10>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(ExpMinusOne, expm1, isFloating<A>())

cudaError_t srtExpMinusOne(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<ExpMinusOne>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Gamma, tgamma, isFloating<A>())

cudaError_t srtGamma(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Gamma>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Hypot, hypot, isFloating<A>())

cudaError_t srtHypot(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Hypot>(a, aDesc, b, bDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Log, log, isFloating<A>())

cudaError_t srtLog(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Log>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(LogOnePlus, log1p, isFloating<A>())

cudaError_t srtLogOnePlus(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<LogOnePlus>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Log2, log2, isFloating<A>())

cudaError_t srtLog2(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Log2>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Log10, log10, isFloating<A>())

cudaError_t srtLog10(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Log10>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(LogGamma, lgamma, isFloating<A>())

cudaError_t srtLogGamma(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<LogGamma>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Mul, multiply, isNumeric<A>())

cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Mul>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtMulTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Mul>(a, aDesc, element, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op3(MultiplyAdd, multiplyAdd, (isNumeric<A>() && isSame<A,Out>()))

cudaError_t srtMultiplyAdd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* c, const srtTensorDescriptor* pcDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsABC(paDesc, pbDesc, pcDesc, poDesc)
    return select<MultiplyAdd>(a, aDesc, b, bDesc, c, cDesc, out, oDesc, stream);
}

Op3Same(MultiplyAddFlat, multiplyAdd, isNumeric<A>())

cudaError_t srtMultiplyAddFlat(
    srtDataType type,
    const void* a,
    const void* b,
    const void* c,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<MultiplyAddFlat>(type, a, b, c, out, count, stream);
}


Op3SwapBC(MultiplyAddE, multiplyAdd, (isNumeric<A>() && isSame<A,Out>()))

cudaError_t srtMultiplyAddTTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<MultiplyAddE>(a, aDesc, element, b, bDesc, out, oDesc, stream);
}

Op3SwapBCSame(MultiplyAddEFlat, multiplyAdd, isNumeric<A>())

cudaError_t srtMultiplyAddFlatTTE(
    srtDataType type,
    const void* a,
    const void* b,
    const void* element,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return selectTET<MultiplyAddEFlat>(type, a, element, b, out, count, stream);
}

//------------------------------------------------------------------------------
Op1(Neg, neg, (isSignedNumeric<A>() || isComplex<A>()))

cudaError_t srtNeg(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Neg>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Pow, pow, isFloating<A>())

cudaError_t srtPow(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Pow>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtPowTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* exponent,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Pow>(a, aDesc, exponent, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Sigmoid, sigmoid, isFloating<A>())

cudaError_t srtSigmoid(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sigmoid>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Sign, sign, isSignedNumeric<A>())

cudaError_t srtSign(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sign>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Sin, sin, isFloating<A>())

cudaError_t srtSin(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sin>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Sinh, sinh, isFloating<A>())

cudaError_t srtSinh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sinh>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Sqrt, sqrt, isFloating<A>())

cudaError_t srtSqrt(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sqrt>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Squared, squared, isNumeric<A>())

cudaError_t srtSquared(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Squared>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op2(Sub, subtract, isNumeric<A>())

cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return select<Sub>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtSubTE(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* element,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Sub>(a, aDesc, element, out, oDesc, stream);
}

// `true` swaps `a` and `element` when calling `divide`
Op2SwapAB(SubET, subtract, (isNumeric<A>() && isSame<A,Out>()))

cudaError_t srtSubET(
    const void* element,
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<SubET>(a, aDesc, element, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Tan, tan, isFloating<A>())

cudaError_t srtTan(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Tan>(a, aDesc, out, oDesc, stream);
}

//------------------------------------------------------------------------------
Op1(Tanh, tanh, isFloating<A>())

cudaError_t srtTanh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return select<Tanh>(a, aDesc, out, oDesc, stream);
}