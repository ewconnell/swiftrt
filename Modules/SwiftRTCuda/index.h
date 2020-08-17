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

//==============================================================================
/// Logical
/// converts grid, block, thread indexes into a logical position
template<size_t Rank>
struct Logical {
    // initializer
    __device__ __forceinline__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) {
        static_assert(Rank <= 3, "only Rank 1 - 3 are implemented");
        if (Rank == 1) {
            position[0] = blockIdx.x * blockDim.x + threadIdx.x;
        } else if (Rank == 2) {
            position[0] = blockIdx.y * blockDim.y + threadIdx.y;
            position[1] = blockIdx.x * blockDim.x + threadIdx.x;
        } else {
            position[0] = blockIdx.z * blockDim.z + threadIdx.z;
            position[1] = blockIdx.y * blockDim.y + threadIdx.y;
            position[2] = blockIdx.x * blockDim.x + threadIdx.x;
        }
    }

    // subscript
    __device__ __forceinline__ uint32_t operator[](int i) const {
        return position[i];
    }

    private:
        // properties
        uint32_t position[Rank];
};

//==============================================================================
/// Single
/// index used for single element value parameters 
template<int Rank>
struct Single {
    // initializer
    __host__ Single(const TensorDescriptor& tensor) { }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical<Rank>& position) const {
        return position[0] == 0;
    }

    /// linear
    /// - Returns: all positions map to the single value, so always returns 0 
    __device__ __forceinline__ 
    uint32_t linear(const Logical<Rank>& position) const { return 0; }
};

//==============================================================================
/// Flat
/// a flat dense index
template<int Rank>
struct Flat {
    uint32_t shape[1];

    //----------------------------------
    // initializer
    __host__ Flat(const TensorDescriptor& tensor) {
        assert(tensor.count == tensor.spanCount);
        shape[0] = tensor.count;
    }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical<Rank>& position) const {
        return position[0] < shape[0];
    }

    //----------------------------------
    __device__ __forceinline__ 
    uint32_t linear(const Logical<Rank>& position) const {
        return position[0];
    }
};

//==============================================================================
/// Strided
template<int Rank>
struct Strided {
    uint32_t shape[Rank];
    uint32_t strides[Rank];

    //----------------------------------
    // initializer
    __host__ Strided(const TensorDescriptor& tensor) {
        for (int i = 0; i < Rank; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            shape[i] = uint32_t(tensor.shape[i]);
            strides[i] = uint32_t(tensor.strides[i]);
        }
    }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical<Rank>& position) const {
        bool inBounds = position[0] < shape[0];
        #pragma unroll
        for (int i = 1; i < Rank; i++) {
            inBounds = inBounds && position[i] < shape[i];
        }
        return inBounds;
    }

    //----------------------------------
    __device__ __forceinline__ 
    uint32_t linear(const Logical<Rank>& position) const {
        uint32_t index = 0;
        #pragma unroll
        for (int i = 0; i < Rank; i++) {
            index += position[i] * strides[i];
        }
        return index;
    }
};

#endif // __index_h__