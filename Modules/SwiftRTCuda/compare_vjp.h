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
#include "compare_fn.h"
#include "index.h"

//==============================================================================
// vjpMin
template<typename T>
__device__ inline T vjpMin(const T& a, const T& b, const T& c) {
    auto m = lessOrEqual(a, b);
    auto v = T();
    if constexpr (packing<T>::count == 1) {
        if (m) v = c;
    } else if constexpr (packing<T>::count == 2) {
        if (m.x) v.x = c.x;
        if (m.y) v.y = c.y;
    } else if constexpr (packing<T>::count == 4) {
        if (m.x) v.x = c.x;
        if (m.y) v.y = c.y;
        if (m.z) v.z = c.z;
        if (m.w) v.w = c.w;
    }
    return v;
}

// vjpMin with two outputs
template<typename T>
__device__ inline void vjpMin(const T& a, const T& b, const T& c, T& outT, T& outF) {
    if constexpr (packing<T>::count == 1) {
        if (a <= b) {
            outT = c; outF = T();
        } else {
            outT = T(); outF = c;
        }
    } else if constexpr (packing<T>::count == 2) {
        vjpMin(a.x, b.x, c.x, outT.x, outF.x);
        vjpMin(a.y, b.y, c.y, outT.y, outF.y);
    } else if constexpr (packing<T>::count == 4) {
        vjpMin(a.x, b.x, c.x, outT.x, outF.x);
        vjpMin(a.y, b.y, c.y, outT.y, outF.y);
        vjpMin(a.z, b.z, c.z, outT.z, outF.z);
        vjpMin(a.w, b.w, c.w, outT.w, outF.w);
    }
}

//==============================================================================
// vjpMax
template<typename T>
__device__ inline T vjpMax(const T& a, const T& b, const T& c) {
    auto m = greaterOrEqual(a, b);
    auto v = T();
    if constexpr (packing<T>::count == 1) {
        if (m) v = c;
    } else if constexpr (packing<T>::count == 2) {
        if (m.x) v.x = c.x;
        if (m.y) v.y = c.y;
    } else if constexpr (packing<T>::count == 4) {
        if (m.x) v.x = c.x;
        if (m.y) v.y = c.y;
        if (m.z) v.z = c.z;
        if (m.w) v.w = c.w;
    }
    return v;
}

// vjpMin with two outputs
template<typename T>
__device__ inline void vjpMax(const T& a, const T& b, const T& c, T& outT, T& outF) {
    if constexpr (packing<T>::count == 1) {
        if (a >= b) {
            outT = c; outF = T();
        } else {
            outT = T(); outF = c;
        }
    } else if constexpr (packing<T>::count == 2) {
        vjpMax(a.x, b.x, c.x, outT.x, outF.x);
        vjpMax(a.y, b.y, c.y, outT.y, outF.y);
    } else if constexpr (packing<T>::count == 4) {
        vjpMax(a.x, b.x, c.x, outT.x, outF.x);
        vjpMax(a.y, b.y, c.y, outT.y, outF.y);
        vjpMax(a.z, b.z, c.z, outT.z, outF.z);
        vjpMax(a.w, b.w, c.w, outT.w, outF.w);
    }
}
