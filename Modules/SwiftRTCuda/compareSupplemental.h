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
#ifndef compareSupplemental_h
#define compareSupplemental_h

#include <vector_types.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
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
// __device__ inline uchar2 operator==(const __half2& a, const __half2& b) {
//     uchar2 out;
//     out.x = a.x == b.x;
//     out.y = a.y == b.y;
//     return out;    
// }

__device__ inline bool2 equalElements(const __nv_bfloat162& a, const __nv_bfloat162& b) {
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


#endif // compareSupplemental_h