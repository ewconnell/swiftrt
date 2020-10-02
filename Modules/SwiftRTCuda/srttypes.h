//******************************************************************************
// Copyright 2020 Google LLC
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
#include <stdint.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

//==============================================================================
// half precision real types
typedef __half  float16;
typedef __half2 float162;

typedef __nv_bfloat16  bfloat16;
typedef __nv_bfloat162 bfloat162;

//==============================================================================
// supplemental logical types
struct bool2 {
    bool b0, b1;
    bool2(bool v0, bool v1) { b0 = v0; b1 = v1; }
};

struct bool4 {
    bool b0, b1, b2, b3;
    bool4(bool v0, bool v1, bool v2, bool v3) { b0 = v0; b1 = v1; b2 = v2; b3 = v3; }
    bool4(unsigned v) { *this = v; }
};
