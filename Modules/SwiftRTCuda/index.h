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

#include "kernelHelpers.h"

//------------------------------------------------------------------------------
// Single
struct Single {
    __device__ inline Single(uint32_t start, uint32_t gridstep, uint32_t count) {}
    __device__ inline uint32_t next() const { return 0; }

    // repeats
    __device__ inline uint32_t pos() { return 0; }
    __device__ inline void setEnd() { }
    __device__ inline bool isEnd(uint32_t pos) const { return false; }
};

//------------------------------------------------------------------------------
// Flat
// a flat dense index
struct Flat {
    __device__ inline Flat(uint32_t start, uint32_t step, uint32_t count) {
        _pos   = start;
        _step = step;
        _end  = count;
    }

    __device__ inline uint32_t pos() { return _pos; }
    __device__ inline uint32_t next() { _pos += _step; return _pos; }
    __device__ inline bool isEnd(uint32_t index) const { return index >= _end; }

    private:
        uint32_t _pos;
        uint32_t _step;
        uint32_t _end;
};


//------------------------------------------------------------------------------
// Index1
template<size_t Rank>
struct StridedIndex {
    const uint32_t shape[Rank];
    const uint32_t strides[Rank];

    // initializer
    __host__ StridedIndex(const srtTensorDescriptor& tensor) {
        for (int i = 0; i < Rank; ++i) {
            assert(tensor.shape[i] <= UINT32_MAX && tensor.strides[i] <= UINT32_MAX);
            shape[i] = uint32_t(tensor.shape[i]);
            strides[i] = uint32_t(tensor.strides[i]);
        }
    }

    __device__ uint32_t linear(dim3 pos) {
        return 0;
    }
};

#endif // __index_h__