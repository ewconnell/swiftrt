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
#ifndef kernels_h
#define kernels_h

#include "index.h"

//==============================================================================
// kernels
//==============================================================================

//------------------------------------------------------------------------------
// tensorA
template<typename Op, typename IndexA, typename IndexO>
__global__ void mapA(
    const typename Op::In* __restrict__ a, const IndexA indexA,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia]);
    }
}

//------------------------------------------------------------------------------
// tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
__global__ void mapAB(
    const typename Op::In* __restrict__ a, const IndexA indexA,
    const typename Op::In* __restrict__ b, const IndexB indexB,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int ib = indexB.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia], b[ib]);
    }
}

//------------------------------------------------------------------------------
// tensorA Scalar
template<typename Op, typename Scalar, typename IndexA, typename IndexO>
__global__ void mapAScalar(
    const typename Op::In* __restrict__ a, const IndexA indexA,
    Scalar value,
    typename Op::Out* __restrict__ out, const IndexO indexO
) {
    auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        int ia = indexA.linear(position);
        int io = indexO.linear(position);
        out[io] = Op::op(a[ia], value);
    }
}

#endif // kernels_h
