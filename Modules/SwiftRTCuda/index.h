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
#include <assert.h>
#include "srt_cdefs.h"

//==============================================================================
// TensorDescriptor
// C++ enhanced wrapper
struct TensorDescriptor: srtTensorDescriptor {
    inline bool isDense() const { return count == spanCount; }
    inline bool isStrided() const { return !isDense(); }
    inline bool isSingle() const { return spanCount == 1; }
};

static_assert(sizeof(TensorDescriptor) == sizeof(srtTensorDescriptor),
    "TensorDescriptor is a c++ wrapper and cannot contain additional members");


// statically cast types from C interface to c++ type
#define Cast2TensorDescriptorsA(pa, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po); \

#define Cast2TensorDescriptorsAB(pa, pb, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po); \

#define Cast2TensorDescriptorsABC(pa, pb, pc, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po); \

#define Cast2TensorDescriptorsABCOO(pa, pb, pc, po0, po1) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& o0Desc = static_cast<const TensorDescriptor&>(*po0); \
const TensorDescriptor& o1Desc = static_cast<const TensorDescriptor&>(*po1); \

#define Cast2TensorDescriptorsAECOO(pa, pc, po0, po1) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& o0Desc = static_cast<const TensorDescriptor&>(*po0); \
const TensorDescriptor& o1Desc = static_cast<const TensorDescriptor&>(*po1); \

//==============================================================================
/// Logical
/// converts grid, block, thread indexes into a logical position
template<size_t Rank> struct LogicalBase {
    uint32_t position[Rank];

    __device__ __forceinline__ uint32_t operator[](int i) const {
        return position[i];
    }
};

template<size_t Rank> struct Logical { };
template<> struct Logical<1> : LogicalBase<1>
{
    __device__ __forceinline__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) : LogicalBase() {
        position[0] = blockIdx.x * blockDim.x + threadIdx.x;
    }
};

template<> struct Logical<2> : LogicalBase<2>
{
    __device__ __forceinline__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) : LogicalBase() {
        position[0] = blockIdx.y * blockDim.y + threadIdx.y;
        position[1] = blockIdx.x * blockDim.x + threadIdx.x;
    }
};

template<> struct Logical<3> : LogicalBase<3>
{
    __device__ __forceinline__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) : LogicalBase() {
        position[0] = blockIdx.z * blockDim.z + threadIdx.z;
        position[1] = blockIdx.y * blockDim.y + threadIdx.y;
        position[2] = blockIdx.x * blockDim.x + threadIdx.x;
    }
};

//==============================================================================
/// Single
/// index used for single element value parameters 
struct Single {
    static const int Rank = 1;
    typedef Logical<1> Logical;

    // initializer
    __host__ Single(const TensorDescriptor& tensor) { }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical& position) const {
        return position[0] == 0;
    }

    /// linear
    /// - Returns: all positions map to the single value, so always returns 0 
    __device__ __forceinline__ 
    uint32_t linear(const Logical& position) const { return 0; }

    __device__ __forceinline__ 
    uint32_t sequence(const Logical& position) const {
        return position[0];
    }
};

//==============================================================================
/// Flat
/// a flat dense 1D index
struct Flat {
    // types
    static const int Rank = 1;
    typedef Logical<Rank> Logical;

    // properties
    uint32_t count;

    //----------------------------------
    // initializer
    __host__ Flat(const TensorDescriptor& tensor) {
        assert(tensor.count == tensor.spanCount);
        count = tensor.count;
    }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical& position) const {
        return position[0] < count;
    }

    //----------------------------------
    __device__ __forceinline__ 
    uint32_t linear(const Logical& position) const {
        return position[0];
    }

    //--------------------------------------------------------------------------
    // the logical sequence position
    __device__ __forceinline__ 
    uint32_t sequence(const Logical& position) const {
        return position[0];
    }
};

//==============================================================================
/// Strided
template<int _Rank>
struct Strided {
    // types
    static const int Rank = _Rank;
    typedef Logical<Rank> Logical;

    // properties
    uint32_t count;
    uint32_t shape[Rank];
    uint32_t strides[Rank];

    //--------------------------------------------------------------------------
    // initializer
    __host__ Strided(const TensorDescriptor& tensor) {
        count = tensor.count;
        for (int i = 0; i < Rank; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            shape[i] = uint32_t(tensor.shape[i]);
            strides[i] = uint32_t(tensor.strides[i]);
        }
    }

    //--------------------------------------------------------------------------
    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __device__ __forceinline__ bool isInBounds(const Logical& position) const {
        bool inBounds = position[0] < shape[0];
        #pragma unroll
        for (int i = 1; i < Rank; i++) {
            inBounds = inBounds && position[i] < shape[i];
        }
        return inBounds;
    }

    //--------------------------------------------------------------------------
    // the linear buffer position
    __device__ __forceinline__ 
    uint32_t linear(const Logical& position) const {
        uint32_t index = 0;
        #pragma unroll
        for (int i = 0; i < Rank; i++) {
            index += position[i] * strides[i];
        }
        return index;
    }
};

//==============================================================================
/// StridedSeq
/// used to calculate strided indexes and sequence positions
/// to support generators
template<int R>
struct StridedSeq: Strided<R> {
    // properties
    uint32_t logicalStrides[R];

    //--------------------------------------------------------------------------
    // initializer
    __host__ StridedSeq(const TensorDescriptor& tensor) : Strided<R>(tensor) {
        for (int i = 0; i < R; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            logicalStrides[i] = uint32_t(tensor.logicalStrides[i]);
        }
    }

    //--------------------------------------------------------------------------
    // the logical sequence position
    __device__ __forceinline__  
    uint32_t sequence(const typename Strided<R>::Logical& position) const {
        uint32_t index = 0;
        #pragma unroll
        for (int i = 0; i < R; i++) {
            index += position[i] * logicalStrides[i];
        }
        return index;
    }
};

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// divideRoundingUp
inline int divideRoundingUp(int num, int divisor) {
    return (num + divisor - 1) / divisor;
}

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic

//--------------------------------------
// tile selection
template<unsigned Rank>
inline dim3 tileSize(const TensorDescriptor& oDesc) {
    static_assert(Rank <= 3, "not implemented");
}

template<> inline dim3 tileSize<1>(const TensorDescriptor& oDesc) {
    return oDesc.count >= 1024 ? dim3(1024) : dim3(32);
}

template<> inline dim3 tileSize<2>(const TensorDescriptor& oDesc) {
    return dim3(16, 16);
}

template<> inline dim3 tileSize<3>(const TensorDescriptor& oDesc) {
    return dim3(16, 8, 8);
}

inline dim3 tileSize(int count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

//--------------------------------------
// grid selection
template<unsigned Rank>
inline dim3 gridSize(const TensorDescriptor& oDesc, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
}

template<>
inline dim3 gridSize<1>(const TensorDescriptor& oDesc, const dim3& tile) {
    return (oDesc.count + tile.x - 1) / tile.x;
}

template<>
inline dim3 gridSize<2>(const TensorDescriptor& oDesc, const dim3& tile) {
    return dim3(divideRoundingUp(oDesc.shape[0], tile.y), 
                divideRoundingUp(oDesc.shape[1], tile.x));
}

template<>
inline dim3 gridSize<3>(const TensorDescriptor& oDesc, const dim3& tile) {
    return dim3(divideRoundingUp(oDesc.shape[0], tile.z), 
                divideRoundingUp(oDesc.shape[1], tile.y), 
                divideRoundingUp(oDesc.shape[2], tile.x));
}
