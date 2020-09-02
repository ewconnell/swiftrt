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
#ifndef dispatchHelpers_h
#define dispatchHelpers_h

#include <stdint.h>
#include <cuda.h>
#include <type_traits>
#include "kernels.h"
#include "complex.h"

//==============================================================================
// supplemental logical types
//==============================================================================
struct bool2 {
    bool b0, b1;
    bool2(bool v0, bool v1) { b0 = v0; b1 = v1; }
};

struct bool4 {
    bool b0, b1, b2, b3;
    bool4(bool v0, bool v1, bool v2, bool v3) { b0 = v0; b1 = v1; b2 = v2; b3 = v3; }
    bool4(unsigned v) { *this = v; }
};

//==============================================================================
// OpBase operator base to express input and out types
template<typename _In, typename _Out> struct OpBase {
    typedef _In In;
    typedef _Out Out;
};

//------------------------------------------------------------------------------
// used for casting between gpu simd types and uint32_t
#define UINT_CREF(_v) reinterpret_cast<const unsigned&>(_v)
#define CAST(type, _v) (*reinterpret_cast<const type*>(&(_v)))

//------------------------------------------------------------------------------
/// fillWord
/// packs elements of size T into a 32 bit word
template<typename T>
inline uint32_t fillWord(const void* pValue) {
    static_assert(sizeof(T) <= sizeof(uint32_t), "T cannot be larger than return type");
    static_assert(std::is_integral<T>::value, "T must be an integral type");
    uint32_t value = uint32_t(*static_cast<const T*>(pValue));
    uint32_t out = value;
    #pragma unroll
    for (int i = 0; i < sizeof(uint32_t) / sizeof(T); ++i) {
        out = (out << sizeof(T) * 8) | value;
    }
    return out;
}

//==============================================================================
// kernel helpers
#define GRID_LOOP(i, n) \
  for (unsigned i = (blockIdx.x * blockDim.x + threadIdx.x); i < (n); \
       i += blockDim.x * gridDim.x)

// shiftDownRoundingUp
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
// dynamic dispatch functions
//==============================================================================

//------------------------------------------------------------------------------
/// flattened tensorA
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapA<Op,Flat,Flat><<<grid, tile, 0, stream>>>(a, Flat(aDesc), out, Flat(oDesc));

    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// flattened tensorA Scalar
template<typename Op, typename Scalar>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    mapAScalar<Op,Scalar,Flat,Flat>
        <<<grid, tile, 0, stream>>>(a, Flat(aDesc), value, out, Flat(oDesc));

    return cudaSuccess;
}

//------------------------------------------------------------------------------
/// flattened tensorA tensorB
template<typename Op>
static cudaError_t flattened(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream,
    int shiftCount = 0
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    const typename Op::In* b = static_cast<const typename Op::In*>(pB);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    int packedCount = shiftDownRoundingUp(oDesc.count, shiftCount);
    dim3 tile = tileSize(packedCount);
    dim3 grid = gridSize<1>(oDesc, tile);

    if (bDesc.isSingle()) {
        mapAB<Op,Flat,Single,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Single(bDesc), out, Flat(oDesc));

    } else if (aDesc.isSingle()) {
        mapAB<Op,Single,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Single(aDesc), b, Flat(bDesc), out, Flat(oDesc));
    } else {
        mapAB<Op,Flat,Flat,Flat><<<grid, tile, 0, stream>>>
            (a, Flat(aDesc), b, Flat(bDesc), out, Flat(oDesc));
    }
    return cudaSuccess;
}

//==============================================================================
// initIndex tensorA
template<typename Op, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapA<Op,IndexA,IndexO>
        <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), out, IndexO(oDesc));

    return cudaSuccess;
}

// initIndex tensorA Scalar
template<typename Op, typename Scalar, typename IndexA, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc, 
    Scalar value,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    mapAScalar<Op,Scalar,IndexA,IndexO>
        <<<grid, tile, 0, stream>>>(a, IndexA(aDesc), value, out, IndexO(oDesc));

    return cudaSuccess;
}

// initIndex tensorA tensorB
template<typename Op, typename IndexA, typename IndexB, typename IndexO>
static cudaError_t initIndex(
    const void* pA, const TensorDescriptor& aDesc,
    const void* pB, const TensorDescriptor& bDesc,
    void* pOut, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    const typename Op::In* a = static_cast<const typename Op::In*>(pA);
    const typename Op::In* b = static_cast<const typename Op::In*>(pB);
    typename Op::Out* out = static_cast<typename Op::Out*>(pOut);

    // get tile and grid size for launch
    dim3 tile = tileSize<IndexO::Rank>(oDesc);
    dim3 grid = gridSize<IndexO::Rank>(oDesc, tile);

    IndexA indexA = IndexA(aDesc);
    IndexB indexB = IndexB(bDesc);
     
    mapAB<Op,IndexA,IndexB,IndexO><<<grid, tile, 0, stream>>>
        (a, indexA, b, indexB, out, IndexO(oDesc));

    return cudaSuccess;
}

//==============================================================================
// selectRank tensorA
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order);
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>>(a, aDesc, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>>(a, aDesc, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA Scalar
template<typename Op, typename Scalar>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    Scalar value,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == oDesc.order);
    // must be same data type and rank, and output is dense
    assert(aDesc.type == oDesc.type && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Scalar,Strided<1>,Strided<1>>(a, aDesc, value, out, oDesc, stream);
    case 2: return initIndex<Op,Scalar,Strided<2>,Strided<2>>(a, aDesc, value, out, oDesc, stream);
    case 3: return initIndex<Op,Scalar,Strided<3>,Strided<3>>(a, aDesc, value, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectRank tensorA tensorB
template<typename Op>
static cudaError_t selectRank(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // for now require the same order
    // TODO: maybe allow simultaneous reordering of elements??
    assert(aDesc.order == bDesc.order && aDesc.order == oDesc.order);

    // must be same data type and rank, and output is dense
    assert(aDesc.type == bDesc.type && aDesc.type == oDesc.type &&
        aDesc.rank == bDesc.rank && aDesc.rank == oDesc.rank);

    switch (oDesc.rank) {
    case 1: return initIndex<Op,Strided<1>,Strided<1>,Strided<1>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 2: return initIndex<Op,Strided<2>,Strided<2>,Strided<2>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    case 3: return initIndex<Op,Strided<3>,Strided<3>,Strided<3>>
        (a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectFloatingStrided tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectRank<Op<float>>(a, aDesc, out, oDesc, stream);
    case real16F:  return selectRank<Op<__half>>(a, aDesc, out, oDesc, stream);
    case real16BF: return selectRank<Op<__nv_bfloat16>>(a, aDesc, out, oDesc, stream);
    case real64F:  return selectRank<Op<double>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectFloatingStrided tensorA Scalar
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op, typename Scalar>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    Scalar value,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectRank<Op<float>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case real16F:  return selectRank<Op<__half>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case real16BF: return selectRank<Op<__nv_bfloat16>,Scalar>(a, aDesc, value, out, oDesc, stream);
    case real64F:  return selectRank<Op<double>,Scalar>(a, aDesc, value, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectFloatingStrided tensorA tensorB
template<template<typename T> class Op>
static cudaError_t selectFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real32F:  return selectRank<Op<float>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16F:  return selectRank<Op<__half>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real16BF: return selectRank<Op<__nv_bfloat16>>(a, aDesc, b, bDesc, out, oDesc, stream);
    case real64F:  return selectRank<Op<double>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectComplexStrided tensorA tensorB
template<template<typename T> class Op>
static cudaError_t selectComplexStrided(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case complex32F:  return selectRank<Op<Complex<float>>>(a, aDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

// selectComplexStrided tensorA tensorB
template<template<typename T> class Op>
static cudaError_t selectComplexStrided(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case complex32F:  return selectRank<Op<Complex<float>>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectIntFloatingStrided tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectIntFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // check float types first
    auto status = selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        switch(oDesc.type) {
        case real32I: return selectRank<Op<int32_t>>(a, aDesc, out, oDesc, stream);
        case real8I:  return selectRank<Op<int8_t>>(a, aDesc, out, oDesc, stream);
        case real8U:  return selectRank<Op<uint8_t>>(a, aDesc, out, oDesc, stream);
        case real16I: return selectRank<Op<int16_t>>(a, aDesc, out, oDesc, stream);
        case real16U: return selectRank<Op<uint16_t>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        return status;
    }
}

// selectIntFloatingStrided tensorA tensorB
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectIntFloatingStrided(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // check float types first
    auto status = selectFloatingStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        switch(oDesc.type) {
        case real32I: return selectRank<Op<int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real8I:  return selectRank<Op<int8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real8U:  return selectRank<Op<uint8_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16I: return selectRank<Op<int16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16U: return selectRank<Op<uint16_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    } else {
        return status;
    }
}

// selectBoolStrided tensorA tensorB
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectBoolStrided(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    switch(oDesc.type) {
    case real8U: return selectRank<Op<bool>>(a, aDesc, b, bDesc, out, oDesc, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// selectFloating tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloating(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<float>>(a, aDesc, out, oDesc, stream);
        case real16F:  return flattened<Op<__half2>>(a, aDesc, out, oDesc, stream, 1);
        case real16BF: return flattened<Op<__nv_bfloat162>>(a, aDesc, out, oDesc, stream, 1);
        case real64F:  return flattened<Op<double>>(a, aDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// selectFloating tensorA Scalar
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op, typename Scalar>
static cudaError_t selectFloating(
    const void* a, const TensorDescriptor& aDesc,
    Scalar value,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, value, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<float>,Scalar>(a, aDesc, value, out, oDesc, stream);
        case real16F:  return flattened<Op<__half2>,Scalar>(a, aDesc, value, out, oDesc, stream, 1);
        case real16BF: return flattened<Op<__nv_bfloat162>,Scalar>(a, aDesc, value, out, oDesc, stream, 1);
        case real64F:  return flattened<Op<double>,Scalar>(a, aDesc, value, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

// selectFloating tensorA tensorB
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectFloating(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.isStrided() || bDesc.isStrided()) {
        return selectFloatingStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
    } else {
        switch(oDesc.type) {
        case real32F:  return flattened<Op<float>>(a, aDesc, b, bDesc, out, oDesc, stream);
        case real16F:  return flattened<Op<__half2>>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case real16BF: return flattened<Op<__nv_bfloat162>>(a, aDesc, b, bDesc, out, oDesc, stream, 1);
        case real64F:  return flattened<Op<double>>(a, aDesc, b, bDesc, out, oDesc, stream);
        default: return cudaErrorNotSupported;
        }
    }
}

//==============================================================================
// selectIntFloating tensorA
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectIntFloating(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    auto status = selectFloating<Op>(a, aDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        if (aDesc.isStrided()) {
            return selectIntFloatingStrided<Op>(a, aDesc, out, oDesc, stream);
        } else {
            switch(oDesc.type) {
            case real32I: return flattened<Op<int32_t>>(a, aDesc, out, oDesc, stream);
            case real8U:  return flattened<Op<uchar4>>(a, aDesc, out, oDesc, stream, 2);
            case real8I:  return flattened<Op<char4>>(a, aDesc, out, oDesc, stream, 2);
            case real16U: return flattened<Op<short2>>(a, aDesc, out, oDesc, stream, 1);
            case real16I: return flattened<Op<short2>>(a, aDesc, out, oDesc, stream, 1);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
    }
}

// selectIntFloating tensorA tensorB
// converts from dynamic to static type and delegates for stride selection
template<template<typename T> class Op>
static cudaError_t selectIntFloating(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // first check if it's a floating type
    auto status = selectFloating<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        if (aDesc.isStrided() || bDesc.isStrided()) {
            return selectIntFloatingStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
        } else {
            switch(oDesc.type) {
            case real32I:  return flattened<Op<int32_t>>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
    }
}

//==============================================================================
// selectNumeric tensorA tensorB
// converts from dynamic Int, Float, Complex to static type
template<template<typename T> class Op>
static cudaError_t selectNumeric(
    const void* a, const TensorDescriptor& aDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // first check if it's an integer or floating type
    auto status = selectIntFloating<Op>(a, aDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        if (aDesc.isStrided()) {
            return selectComplexStrided<Op>(a, aDesc, out, oDesc, stream);
            return cudaErrorNotSupported;
        } else {
            // switch on complex type
            switch(oDesc.type) {
            case complex32F: return flattened<Op<Complex<float>>>(a, aDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
    }
}

// selectNumeric tensorA tensorB
// converts from dynamic Int, Float, Complex to static type
template<template<typename T> class Op>
static cudaError_t selectNumeric(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    // first check if it's an integer or floating type
    auto status = selectIntFloating<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
    if (status == cudaErrorNotSupported) {
        if (aDesc.isStrided() || bDesc.isStrided()) {
            return selectComplexStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
            return cudaErrorNotSupported;
        } else {
            // switch on complex type
            switch(oDesc.type) {
            case complex32F: return flattened<Op<Complex<float>>>(a, aDesc, b, bDesc, out, oDesc, stream);
            default: return cudaErrorNotSupported;
            }
        }
    } else {
        return status;
    }
}

//==============================================================================
// selectBool tensorA tensorB
// converts from dynamic to static type.
template<template<typename T> class Op>
static cudaError_t selectBool(
    const void* a, const TensorDescriptor& aDesc,
    const void* b, const TensorDescriptor& bDesc,
    void* out, const TensorDescriptor& oDesc,
    cudaStream_t stream
) {
    if (aDesc.isStrided()) {
        return selectBoolStrided<Op>(a, aDesc, b, bDesc, out, oDesc, stream);
        return cudaErrorNotSupported;
    } else {
        switch(oDesc.type) {
        case boolean: return flattened<Op<bool4>>(a, aDesc, b, bDesc, out, oDesc, stream, 2);
        default: return cudaErrorNotSupported;
        }
    }
}

#endif // dispatchHelpers_h