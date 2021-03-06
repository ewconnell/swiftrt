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
#include <type_traits>

#include "srt_traits.h"
#include "tensor.cuh"

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
/// Logical
/// converts grid, block, thread indexes into a logical position
template<int Rank> struct LogicalBase {
    uint32_t position[Rank];

    __DEVICE_INLINE__ uint32_t operator[](int i) const {
        return position[i];
    }
};

template<int Rank> struct Logical { };
template<> struct Logical<1> : LogicalBase<1> {

    __DEVICE_INLINE__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) : LogicalBase() {
        position[0] = blockIdx.x * blockDim.x + threadIdx.x;
    }
};

template<> struct Logical<2> : LogicalBase<2> {

    __DEVICE_INLINE__ Logical(
        const uint3& blockIdx,
        const dim3& blockDim,
        const uint3& threadIdx
    ) : LogicalBase() {
        position[0] = blockIdx.y * blockDim.y + threadIdx.y;
        position[1] = blockIdx.x * blockDim.x + threadIdx.x;
    }
};

template<> struct Logical<3> : LogicalBase<3> {

    __DEVICE_INLINE__ Logical(
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
/// Constant
/// index used to generate a constant value
template<typename T, int Rank>
struct Constant {
    typedef Logical<Rank> Logical;
    const T value;

    // initializer
    __HOST_INLINE__ Constant(const T& v) : value(v) { }

    //----------------------------------
    // get
    __DEVICE_INLINE__ T operator[](const Logical& pos) const {
        return value;
    }

    __DEVICE_INLINE__ T operator[](uint32_t index) const {
        return value;
    }

    //--------------------------------------------------------------------------
    // the linear buffer position
    __DEVICE_INLINE__ uint32_t linear(const Logical& position) const {
        return 0;
    }

    __DEVICE_INLINE__ uint32_t sequence(const Logical& position) const {
        return position[0];
    }
};

//==============================================================================
/// Flat
/// a random access iterator for a dense 1D buffer
template<typename PointerT>
struct Flat {
    // type of buffer element
    typedef typename std::remove_pointer<PointerT>::type T;

    // type of logical position
    static const int Rank = 1;
    typedef Logical<Rank> Logical;

    // pointer to element buffer
    PointerT buffer;
    // shape of buffer
    uint32_t count;

    //----------------------------------
    // initializer
    __HOST_INLINE__ Flat(PointerT p, uint32_t _count) {
        buffer = p;
        count = divideRoundingUp(_count, packing<T>::count);
    }

    /// isInBounds
    /// `true` if the given logical position is within the bounds of
    /// the indexed space
    /// - Parameters:
    ///  - position: the logical position to test
    /// - Returns: `true` if the position is within the shape
    __DEVICE_INLINE__ bool isInBounds(const Logical& position) const {
        return position[0] < count;
    }

    //----------------------------------
    // get
    __DEVICE_INLINE__ T operator[](const Logical& pos) const {
        return buffer[pos[0]];
    }

    __DEVICE_INLINE__ T operator[](uint32_t index) const {
        return buffer[index];
    }

    //----------------------------------
    // set
    __DEVICE_INLINE__ T& operator[](const Logical& pos) {
        return buffer[pos[0]];
    }

    __DEVICE_INLINE__ T& operator[](uint32_t index) {
        return buffer[index];
    }

    //--------------------------------------------------------------------------
    // the linear buffer position
    __DEVICE_INLINE__ uint32_t linear(const Logical& position) const {
        return position[0];
    }

    //--------------------------------------------------------------------------
    // the logical sequence position
    __DEVICE_INLINE__ 
    uint32_t sequence(const Logical& position) const {
        return position[0];
    }
};

//==============================================================================
/// Strided
template<typename PointerT, int R>
struct Strided {
    // type of buffer element
    typedef typename std::remove_pointer<PointerT>::type T;

    // types
    static const int Rank = R;
    typedef Logical<Rank> Logical;

    // pointer to element buffer
    PointerT buffer;
    // shape of buffer
    uint32_t shape[Rank];
    // element strides
    uint32_t strides[Rank];

    //--------------------------------------------------------------------------
    // initializer
    __HOST_INLINE__ Strided(PointerT p, const TensorDescriptor& tensor) {
        buffer = p;
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
    __DEVICE_INLINE__ bool isInBounds(const Logical& position) const {
        bool inBounds = position[0] < shape[0];
        #pragma unroll
        for (int i = 1; i < Rank; i++) {
            inBounds = inBounds && position[i] < shape[i];
        }
        return inBounds;
    }

    //--------------------------------------------------------------------------
    // the linear buffer position
    __DEVICE_INLINE__ uint32_t linear(const Logical& position) const {
        uint32_t index = 0;
        #pragma unroll
        for (int i = 0; i < Rank; i++) {
            index += position[i] * strides[i];
        }
        return index;
    }

    //----------------------------------
    // get
    __DEVICE_INLINE__ T operator[](const Logical& pos) const {
        return buffer[linear(pos)];
    }

    __DEVICE_INLINE__ T operator[](uint32_t index) const {
        return buffer[index];
    }

    //----------------------------------
    // set
    __DEVICE_INLINE__ T& operator[](const Logical& pos) {
        return buffer[linear(pos)];
    }

    __DEVICE_INLINE__ T& operator[](uint32_t index) {
        return buffer[index];
    }
};

//==============================================================================
/// StridedSeq
/// used to calculate strided indexes and sequence positions
/// to support generators
template<typename PointerT, int R>
struct StridedSeq: Strided<PointerT, R> {
    // properties
    typedef typename Strided<PointerT, R>::Logical Logical;
    uint32_t logicalStrides[R];
    uint32_t count;

    //--------------------------------------------------------------------------
    // initializer
    __HOST_INLINE__ StridedSeq(PointerT p, const TensorDescriptor& tensor) :
        Strided<PointerT, R>(p, tensor)
    {
        count = tensor.count;
        for (int i = 0; i < R; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            logicalStrides[i] = uint32_t(tensor.logicalStrides[i]);
        }
    }

    //--------------------------------------------------------------------------
    // the logical sequence position
    __DEVICE_INLINE__ uint32_t sequence(const Logical& position) const {
        uint32_t index = 0;
        #pragma unroll
        for (int i = 0; i < R; i++) {
            index += position[i] * logicalStrides[i];
        }
        return index;
    }
};

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic

//--------------------------------------
// tile selection
template<int Rank>
inline dim3 tileSize(const uint32_t* shape) {
    static_assert(Rank <= 3, "not implemented");
}

template<> inline dim3 tileSize<1>(const uint32_t* shape) {
    return shape[0] >= 1024 ? dim3(1024) : dim3(32);
}

inline dim3 tileSize(uint32_t count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

template<> inline dim3 tileSize<2>(const uint32_t* shape) {
    return dim3(16, 16);
}

template<> inline dim3 tileSize<3>(const uint32_t* shape) {
    return dim3(16, 8, 8);
}

//--------------------------------------
// grid selection
template<int Rank>
inline dim3 gridSize(const uint32_t* shape, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
}

template<>
inline dim3 gridSize<1>(const uint32_t* shape, const dim3& tile) {
    return (shape[0] + tile.x - 1) / tile.x;
}

inline dim3 gridSize(uint32_t count, const dim3& tile) {
    return (count + tile.x - 1) / tile.x;
}

template<>
inline dim3 gridSize<2>(const uint32_t* shape, const dim3& tile) {
    return dim3(divideRoundingUp(shape[1], tile.x),
                divideRoundingUp(shape[0], tile.y));
}

template<>
inline dim3 gridSize<3>(const uint32_t* shape, const dim3& tile) {
    return dim3(divideRoundingUp(shape[2], tile.x), 
                divideRoundingUp(shape[1], tile.y), 
                divideRoundingUp(shape[0], tile.z));
}
