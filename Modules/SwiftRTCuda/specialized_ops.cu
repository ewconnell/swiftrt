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
#include "specialized_api.h"
#include "complex.cuh"
#include "math_fn.cuh"
#include "iterators.cuh"
#include "tensor.cuh"

#include "type_name.hpp"
#include <iostream>       // std::cout

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//==============================================================================
// Julia Set

// tensorA Element
template<typename IterA, typename IterO>
__global__ void mapJulia(
    IterA iterA,
    IterO iterO,
    const typename IterA::T c,
    const typename IterA::T::RealType tolerance2,
    int iterations
) {
    // 0.000416s
    const auto p = typename IterO::Logical(blockIdx, blockDim, threadIdx);
    if (iterO.isInBounds(p)) {
        auto z = iterA[p];
        float d = iterations;
        for (int j = 0; j < iterations; ++j) {
            z = z * z + c;
            if (abs2(z) > tolerance2) {
                d = min(d, float(j));
                break;
            }
        }
        iterO[p] = d;
    }
}

template<typename A>
cudaError_t juliaFlat(
    const void* pA,
    const void* pConstant,
    float tolerance,
    size_t iterations,
    size_t count,
    void* pOut,
    cudaStream_t stream
) {
    typedef typename A::RealType RealType;

    // input is Complex<RealType>
    const A* a = static_cast<const A*>(pA);
    const A c = *static_cast<const A*>(pConstant);
    const RealType tolerance2 = tolerance * tolerance;

    // output (divergence) is RealType
    RealType* out = static_cast<RealType*>(pOut);

    auto iterA = Flat(a, count);
    auto iterO = Flat(out, count);

    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    CudaKernelPreCheck(stream);
    mapJulia<<<grid, tile, 0, stream>>>(iterA, iterO, c, tolerance2, iterations);
    return CudaKernelPostCheck(stream);
}

cudaError_t srtJuliaFlat(
    srtDataType type,
    const void* a,
    const void* constant,
    float tolerance,
    size_t iterations,
    size_t count,
    void* out,
    cudaStream_t stream
) {
    switch (type) {
    case complex16F: return juliaFlat<Complex<float16>>(a, constant, tolerance, iterations, count, out, stream);
    case complex32F: return juliaFlat<Complex<float>>(a, constant, tolerance, iterations, count, out, stream);
    default: return cudaErrorNotSupported;
    }
}

//==============================================================================
// Mandelbrot Set

// tensorA Element
template<typename IterA, typename IterO>
__global__ void mapMandelbrot(
    IterA iterA,
    IterO iterO,
    const typename IterA::T::RealType tolerance2,
    int iterations
) {
    // 0.00111s
    const auto p = typename IterO::Logical(blockIdx, blockDim, threadIdx);
    if (iterO.isInBounds(p)) {
        const auto x = iterA[p];
        auto z = x;
        float d = iterations;
        for (int j = 1; j < iterations; ++j) {
            z = z * z + x;
            if (abs2(z) > tolerance2) {
                d = min(d, float(j));
                break;
            }
        }
        iterO[p] = d;
    }
}

template<typename A>
cudaError_t mandelbrotFlat(
    const void* pA,
    float tolerance,
    size_t iterations,
    size_t count,
    void* pOut,
    cudaStream_t stream
) {
    typedef typename A::RealType RealType;

    // input is Complex<RealType>
    const A* a = static_cast<const A*>(pA);
    const RealType tolerance2 = tolerance * tolerance;

    // output (divergence) is RealType
    RealType* out = static_cast<RealType*>(pOut);

    auto iterA = Flat(a, count);
    auto iterO = Flat(out, count);

    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    CudaKernelPreCheck(stream);
    mapMandelbrot<<<grid, tile, 0, stream>>>(iterA, iterO, tolerance2, iterations);
    return CudaKernelPostCheck(stream);
}

cudaError_t srtMandelbrotFlat(
    srtDataType type,
    const void* a,
    float tolerance,
    size_t iterations,
    size_t count,
    void* out,
    cudaStream_t stream
) {
    switch (type) {
    case complex16F: return mandelbrotFlat<Complex<float16>>(a, tolerance, iterations, count, out, stream);
    case complex32F: return mandelbrotFlat<Complex<float>>(a, tolerance, iterations, count, out, stream);
    default: return cudaErrorNotSupported;
    }
}
