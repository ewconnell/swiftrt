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
#include "srt_traits.cuh"
#include "copy_api.h"
#include "float16.cuh"
#include "bfloat16.cuh"
#include "tensor.cuh"
#include "compare_fn.cuh"

//==============================================================================
// supplemental functions to give names to operator types
// so that ops can reference functions in a uniform way
//==============================================================================

template<typename A, typename Out>
__DEVICE_INLINE__ Out cast(const A& a) { return a; }

//--------------------------------------
// double
template<>
__DEVICE_INLINE__ Complex<float16> cast(const double& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<bfloat16> cast(const double& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<float> cast(const double& a) { return float(a); }

//--------------------------------------
// int8_t
template<>
__DEVICE_INLINE__ bfloat16 cast(const int8_t& a) { return float(a); }

//--------------------------------------
// uint8_t
template<>
__DEVICE_INLINE__ bfloat16 cast(const uint8_t& a) { return float(a); }

//--------------------------------------
// int16_t
template<>
__DEVICE_INLINE__ bfloat16 cast(const int16_t& a) { return float(a); }

//--------------------------------------
// uint16_t
template<>
__DEVICE_INLINE__ bfloat16 cast(const uint16_t& a) { return float(a); }

//--------------------------------------
// int32_t
template<>
__DEVICE_INLINE__ bfloat16 cast(const int32_t& a) { return float(a); }

//--------------------------------------
// bool
template<>
__DEVICE_INLINE__ bfloat16 cast(const bool& a) { return a ? 1.0f : 0.0f; }

template<>
__DEVICE_INLINE__ float16 cast(const bool& a) { return a ? 1.0f : 0.0f; }

//--------------------------------------
// float16
template<>
__DEVICE_INLINE__ bool cast(const float16& a) { return a != float16(); }

template<>
__DEVICE_INLINE__ bfloat16 cast(const float16& a) { return float(a); }

template<>
__DEVICE_INLINE__ int8_t cast(const float16& a) { return float(a); }

template<>
__DEVICE_INLINE__ uint8_t cast(const float16& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<bfloat16> cast(const float16& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<float> cast(const float16& a) { return float(a); }

//--------------------------------------
// float162
template<>
__DEVICE_INLINE__ bfloat162 cast(const float162& a) {
    bfloat162 t; t.x = float(a.x); t.y = float(a.y); return t;
}

template<>
__DEVICE_INLINE__ bool2 cast(const float162& a) {
    return notEqual(a, float162());
}

//--------------------------------------
// bfloat16
template<>
__DEVICE_INLINE__ bool cast(const bfloat16& a) { return a != bfloat16(); }

template<>
__DEVICE_INLINE__ float16 cast(const bfloat16& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<float16> cast(const bfloat16& a) { return float(a); }

template<>
__DEVICE_INLINE__ Complex<float> cast(const bfloat16& a) { return float(a); }

//--------------------------------------
// bfloat162
template<>
__DEVICE_INLINE__ float162 cast(const bfloat162& a) {
    float162 t; t.x = float(a.x); t.y = float(a.y); return t;
}

template<>
__DEVICE_INLINE__ bool2 cast(const bfloat162& a) {
    return notEqual(a, bfloat162());
}

// template<>
// __DEVICE_INLINE__ short2 cast(const bfloat162& a) { return make_short2(a.x, a.y); }

//--------------------------------------
// Complex
template<>
__DEVICE_INLINE__ Complex<float16> cast(const Complex<float>& a) {
    return Complex<float16>(a.x, a.y);
}

template<>
__DEVICE_INLINE__ Complex<bfloat16> cast(const Complex<float>& a) {
    return Complex<bfloat16>(a.x, a.y);
}

template<>
__DEVICE_INLINE__ Complex<float> cast(const Complex<float16>& a) {
    return Complex<float>(a.x, a.y);
}

template<>
__DEVICE_INLINE__ Complex<bfloat16> cast(const Complex<float16>& a) {
    return Complex<bfloat16>(float(a.x), float(a.y));
}

template<>
__DEVICE_INLINE__ Complex<float> cast(const Complex<bfloat16>& a) {
    return Complex<float>(a.x, a.y);
}

template<>
__DEVICE_INLINE__ Complex<float16> cast(const Complex<bfloat16>& a) {
    return Complex<float16>(float(a.x), float(a.y));
}

//==============================================================================

template<typename _A, typename _O> struct CastOp {
    typedef _A A; typedef _O Out;
    static_assert(isPacked<A>() == isPacked<Out>(), "packed type mismatch");
    constexpr static bool conforms() {
        return !isComplex<A>() || (isComplex<A>() && isComplex<Out>());
    }
    __DEVICE_INLINE__ void operator()(const A& a, Out& out) const {
        out = cast<A,Out>(a);
    }
    typedef typename packed<A>::type PA;
    typedef typename matching_packed<PA,Out>::type POut;
    typedef CastOp<PA,POut> packed;
};

