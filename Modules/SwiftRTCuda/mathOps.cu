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
#include "mathOps.h"
#include "mathSupplemental.h"

//==============================================================================
// ops
//==============================================================================

#define MATHOP(OpName, name) \
template<typename T> struct OpName: OpBase<T,T> { \
    __device__ inline static T op(const T& a) { return name(a); } \
}; \

#define MATHOP2N(OpName, name) \
template<typename T> struct OpName: OpBase<T,T> { \
    __device__ inline static T op(const T& a, const int n) { return name(a, n); } \
}; \

#define MATHOP2(OpName, name) \
template<typename T> struct OpName: OpBase<T,T> { \
    __device__ inline static T op(const T& a, const T& b) { return name(a, b); } \
}; \

MATHOP2(Add, add)
MATHOP(Abs, abs)
MATHOP(Acos, acos)
MATHOP(Acosh, acosh)
MATHOP(Asin, asin)
MATHOP(Asinh, asinh)
MATHOP(Atan, atan)
MATHOP2(Atan2, atan2)
MATHOP(Atanh, atanh)
MATHOP(Cos, cos)
MATHOP(Cosh, cosh)
MATHOP2(Div, divide)
MATHOP(Erf, erf)
MATHOP(Erfc, erfc)
MATHOP(Exp, exp)
MATHOP(Exp2, exp2)
MATHOP(Exp10, exp10)
MATHOP(ExpMinusOne, expm1)
MATHOP(Gamma, tgamma)
MATHOP2(Hypot, hypot)
MATHOP(Log, log)
MATHOP(LogOnePlus, log1p)
MATHOP(Log2, log2)
MATHOP(Log10, log10)
MATHOP(LogGamma, lgamma)
MATHOP2(Mul, multiply)
MATHOP(Neg, neg)
MATHOP2(Pow, pow)
MATHOP2N(PowN, pow)
MATHOP2N(Root, root)
MATHOP(Sigmoid, sigmoid)
MATHOP(Sign, sign)
MATHOP(Sin, sin)
MATHOP(Sinh, sinh)
MATHOP(Sqrt, sqrt)
MATHOP(Squared, squared)
MATHOP2(Sub, subtract)
MATHOP(Tan, tan)
MATHOP(Tanh, tanh)


//==============================================================================
// Swift importable C interface functions
//==============================================================================

// All types
cudaError_t srtAbs(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectIntFloating<Abs>(a, aDesc, out, oDesc, stream);
}

// Must be promoted types
cudaError_t srtAcos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Acos>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtAcosh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Acosh>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectNumeric<Add>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtAsin(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Asin>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtAsinh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Asinh>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtAtan(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Atan>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtAtan2(
    const void* b, const srtTensorDescriptor* pbDesc,
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    // b comes first
    return selectFloating<Atan2>(b, bDesc, a, aDesc, out, oDesc, stream);
}

cudaError_t srtAtanh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Atanh>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtCos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Cos>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtCosh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Cosh>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectNumeric<Div>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtErf(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Erf>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtErfc(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Erfc>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtExp(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Exp>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtExp2(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Exp2>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtExp10(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Exp10>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtExpMinusOne(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<ExpMinusOne>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtGamma(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Gamma>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtHypot(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectFloating<Hypot>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtLog(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Log>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtLogOnePlus(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<LogOnePlus>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtLog2(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Log2>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtLog10(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Log10>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtLogGamma(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<LogGamma>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectNumeric<Mul>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtNeg(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectNumeric<Neg>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtPow(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectFloating<Pow>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtPowN(
    const void* a, const srtTensorDescriptor* paDesc, long n,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<PowN>(a, aDesc, int(n), out, oDesc, stream);
}

cudaError_t srtRoot(
    const void* a, const srtTensorDescriptor* paDesc, long n,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Root>(a, aDesc, int(n), out, oDesc, stream);
}

cudaError_t srtSigmoid(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Sigmoid>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSign(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectIntFloating<Sign>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSin(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Sin>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSinh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Sinh>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSqrt(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Sqrt>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSquared(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectNumeric<Squared>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* paDesc,
    const void* b, const srtTensorDescriptor* pbDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    Cast2TensorDescriptorsAB(paDesc, pbDesc, poDesc)
    return selectNumeric<Sub>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtTan(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Tan>(a, aDesc, out, oDesc, stream);
}

cudaError_t srtTanh(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectFloating<Tanh>(a, aDesc, out, oDesc, stream);
}
