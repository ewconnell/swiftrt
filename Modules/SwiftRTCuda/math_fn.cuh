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
#include "math_api.cuh"
#include "float16.cuh"
#include "bfloat16.cuh"

//==============================================================================
// supplemental functions to give names to operator types
// so that ops can reference functions in a uniform way
//==============================================================================

template<typename T>
__DEVICE_INLINE__ T add(const T& a, const T& b) { return a + b; }

template<typename T>
__DEVICE_INLINE__ T subtract(const T& a, const T& b) { return a - b; }

template<typename T>
__DEVICE_INLINE__ T multiply(const T& a, const T& b) { return a * b; }

template<typename T>
__DEVICE_INLINE__ T multiplyAdd(const T& a, const T& b, const T& c) {
    return a * b + c;
}

template<typename T>
__DEVICE_INLINE__ T divide(const T& a, const T& b) { return a / b; }

template<typename T>
__DEVICE_INLINE__ T neg(const T& a) { return -a; }

template<typename T>
__DEVICE_INLINE__ T sign(const T& a) { return a < T(0) ? T(-1) : T(1); }

template<typename T>
__DEVICE_INLINE__ T squared(const T& a) { return a * a; }

// this is ugly because bfloats don't have first class support < sm800
template<typename T>
__DEVICE_INLINE__ T sigmoid(const T& a) { return divide(init<T>(1), (add(init<T>(1), exp(neg(a))))); }


