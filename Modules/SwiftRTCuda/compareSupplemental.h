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
#pragma once
#include "dispatchHelpers.h"

//==============================================================================
// supplemental logical functions
//==============================================================================

//------------------------------------------------------------------------------
// andElements 
__device__ inline bool andElements(const bool& a, const bool& b) {
    return a & b;
}

__device__ inline bool4 andElements(const bool4& a, const bool4& b) {
    const unsigned out = UINT_CREF(a) & UINT_CREF(b);
    return CAST(bool4, out);
}

//------------------------------------------------------------------------------
// orElements 
__device__ inline bool orElements(const bool& a, const bool& b) {
    return a | b;
}

__device__ inline bool4 orElements(const bool4& a, const bool4& b) {
    const unsigned out = UINT_CREF(a) | UINT_CREF(b);
    return CAST(bool4, out);
}

//------------------------------------------------------------------------------
// equalElements
__device__ inline bool2 equalElements(const bfloat162& a, const bfloat162& b) {
    return bool2(a.x == b.x, a.y == b.y);    
}

__device__ inline bool4 equalElements(const char4& a, const char4& b) {
    const auto out = __vcmpeq4(UINT_CREF(a), UINT_CREF(b));
    return CAST(bool4, out);
}

template<typename T>
__device__ inline bool equalElements(const T& a, const T& b) {
    return a == b;
}

//------------------------------------------------------------------------------
// notEqualElements
__device__ inline bool2 notEqualElements(const bfloat162& a, const bfloat162& b) {
    return bool2(a.x != b.x, a.y != b.y);    
}

__device__ inline bool4 notEqualElements(const char4& a, const char4& b) {
    const auto out = __vcmpeq4(UINT_CREF(a), UINT_CREF(b)) - 1;
    return CAST(bool4, out);
}

template<typename T>
__device__ inline bool notEqualElements(const T& a, const T& b) {
    return a != b;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool greaterElements(const T& a, const T& b) {
    return a > b;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool greaterOrEqualElements(const T& a, const T& b) {
    return a >= b;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline bool lessElements(const T& a, const T& b) {
    return a < b;
}

//------------------------------------------------------------------------------
__device__ inline bool2 lessOrEqualElements(const bfloat162& a, const bfloat162& b) {
    return bool2(a.x <= b.x, a.y <= b.y);
}

template<typename T>
__device__ inline bool lessOrEqualElements(const T& a, const T& b) {
    return a <= b;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline T minElements(const T& a, const T& b) {
    return a <= b ? a : b;
}

__device__ inline bfloat162 minElements(const bfloat162& a, const bfloat162& b) {
    bfloat162 v;
    v.x = a.x <= b.x ? a.x : b.x;
    v.y = a.y <= b.y ? a.y : b.y;
    return v;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline T maxElements(const T& a, const T& b) {
    return a > b ? a : b;
}

__device__ inline bfloat162 maxElements(const bfloat162& a, const bfloat162& b) {
    bfloat162 v;
    v.x = a.x > b.x ? a.x : b.x;
    v.y = a.y > b.y ? a.y : b.y;
    return v;
}

//------------------------------------------------------------------------------
template<typename T>
__device__ inline T conditionalAssign(const T& a, const T& b, const bool c) {
    return c ? a : b;
}
