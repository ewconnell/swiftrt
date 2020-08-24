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
#ifndef kernelHelpers_h
#define kernelHelpers_h

#include <assert.h>
#include "commonCDefs.h"
#include <vector_types.h>

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

//------------------------------------------------------------------------------
// 
inline unsigned shiftDownRoundingUp(unsigned num, unsigned shift) {
    unsigned count = (num + (1 << shift) - 1) >> shift;
    return count;
}

//------------------------------------------------------------------------------
/// roundUp
// tiles should always be shaped as a power of 2
// TODO: there should be something faster than this
inline unsigned roundUp(unsigned n, unsigned multiple) {
    return (n + multiple - 1) / multiple;
}

//==============================================================================
// grid and tile size placeholders

// *** this is a hack place holder for now. We will eventually do dynamic
// tile selection
template<unsigned Rank>
inline dim3 tileSize(const TensorDescriptor& oDesc) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return oDesc.count >= 1024 ? dim3(1024) : dim3(32);
    if (Rank == 2) return dim3(16, 16);
    if (Rank == 3) return dim3(16, 8, 8);
}

inline dim3 tileSize(int count) {
    return count >= 1024 ? dim3(1024) : dim3(32);
}

template<unsigned Rank>
inline dim3 gridSize(const TensorDescriptor& oDesc, const dim3& tile) {
    static_assert(Rank <= 3, "not implemented");
    if (Rank == 1) return (oDesc.count + tile.x - 1) / tile.x;

    if (Rank == 2) return dim3(roundUp(oDesc.shape[0], tile.y), 
                               roundUp(oDesc.shape[1], tile.x));
    
    if (Rank == 3) return dim3(roundUp(oDesc.shape[0], tile.z), 
                               roundUp(oDesc.shape[1], tile.y), 
                               roundUp(oDesc.shape[2], tile.x));
}

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// threads per block
const unsigned THREADS_PER_BLOCK = 1024;

// number of blocks for threads
inline unsigned BLOCK_COUNT(unsigned n) {
  return ((n) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

#endif // kernelHelpers_h