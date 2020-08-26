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
#include "mathOps.h"
#include "mathSupplemental.h"
#include "dispatchHelpers.h"

//==============================================================================
// ops
//==============================================================================

template<typename T> struct OpBase { typedef T Element; };

#define MATHOP(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
    __device__ inline static T op(const T& a) { return name(a); } \
}; \

#define MATHOP2N(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
    __device__ inline static T op(const T& a, const int n) { return name(a, n); } \
}; \

#define MATHOP2(OpName, name) \
template<typename T> struct OpName: OpBase<T> { \
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
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectAnyPacked<Abs>(x, xDesc, out, oDesc, stream);
}

// Must be promoted types
cudaError_t srtAcos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Acos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAcosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Acosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAdd(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectAnyPacked<Add>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtAsin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Asin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAsinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Asinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Atan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtan2(
    const void* y, const srtTensorDescriptor* yDesc,
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    // y comes first
    return selectFloating<Atan2>(y, yDesc, x, xDesc, out, oDesc, stream);
}

cudaError_t srtAtanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Atanh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCos(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Cos>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtCosh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Cosh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtDiv(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectAnyPacked<Div>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtErf(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Erf>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtErfc(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Erfc>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExp10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Exp10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtExpMinusOne(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<ExpMinusOne>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Gamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtHypot(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Hypot>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtLog(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogOnePlus(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<LogOnePlus>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog2(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log2>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLog10(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Log10>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtLogGamma(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<LogGamma>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtMul(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectAnyPacked<Mul>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtNeg(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectAny<Neg>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtPow(
    const void* x, const srtTensorDescriptor* xDesc,
    const void* y, const srtTensorDescriptor* yDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Pow>(x, xDesc, y, yDesc, out, oDesc, stream);
}

cudaError_t srtPowN(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<PowN>(x, xDesc, int(n), out, oDesc, stream);
}

cudaError_t srtRoot(
    const void* x, const srtTensorDescriptor* xDesc, long n,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Root>(x, xDesc, int(n), out, oDesc, stream);
}

cudaError_t srtSigmoid(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Sigmoid>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSign(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectAny<Sign>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSin(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Sin>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSinh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Sinh>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSqrt(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloatingPacked<Sqrt>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSquared(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Squared>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtSub(
    const void* a, const srtTensorDescriptor* aDesc,
    const void* b, const srtTensorDescriptor* bDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream
) {
    return selectAnyPacked<Sub>(a, aDesc, b, bDesc, out, oDesc, stream);
}

cudaError_t srtTan(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Tan>(x, xDesc, out, oDesc, stream);
}

cudaError_t srtTanh(
    const void* x, const srtTensorDescriptor* xDesc,
    void* out, const srtTensorDescriptor* oDesc,
    cudaStream_t stream)
{
    return selectFloating<Tanh>(x, xDesc, out, oDesc, stream);
}
