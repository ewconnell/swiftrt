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
#if !defined(__index_h__)
#define __index_h__

#include <vector_types.h>
#include "kernelHelpers.h"

//------------------------------------------------------------------------------
// Flat
// a flat dense index
struct Flat {
    __device__ __forceinline__ static uint32_t linearIndex(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
};


//------------------------------------------------------------------------------
// Index1
template<size_t Rank>
struct Strided {
    uint32_t shape[Rank];
    uint32_t strides[Rank];

    // initializer
    __host__ Strided(const TensorDescriptor& tensor) {
        for (int i = 0; i < Rank; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            shape[i] = uint32_t(tensor.shape[i]);
            strides[i] = uint32_t(tensor.strides[i]);
        }
    }

    //--------------------------------------------------------------------------
    /// linearIndex
    __device__ __forceinline__ uint32_t linearIndex(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) const {
        static_assert(Rank <= 3, "only Rank 1 - 3 are implemented");
        if (Rank == 1) {
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            return col * strides[0];

        } else if (Rank == 2) {
            auto row = blockIdx.y * blockDim.y + threadIdx.y;
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            return row * strides[0] + col * strides[1];

        } else {
            auto dep = blockIdx.z * blockDim.z + threadIdx.z;
            auto row = blockIdx.y * blockDim.y + threadIdx.y;
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            return dep * strides[0] + row * strides[1] + col * strides[2];
        }
    }
};

#endif // __index_h__