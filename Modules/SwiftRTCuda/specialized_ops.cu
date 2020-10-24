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
    const typename IterO::T::RealType tolerance2,
    const typename IterO::T c,
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
    const void* pTolerance,
    size_t iterations,
    size_t count,
    void* pOut,
    cudaStream_t stream
) {
    typedef typename A::RealType RT;

    const A* a = static_cast<const A*>(pA);
    A* out = static_cast<A*>(pOut);
    const RT tolerance = *static_cast<const RT*>(pTolerance);
    const A c = *static_cast<const A*>(pConstant);

    auto iterA = Flat(a, count);
    auto iterO = Flat(out, count);
    auto tolerance2 = RT(float(tolerance) * float(tolerance));

    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    mapJulia<<<grid, tile, 0, stream>>>(iterA, iterO, tolerance2, c, iterations);
    return cudaSuccess;
}

cudaError_t srtJuliaFlat(
    srtDataType type,
    const void* a,
    const void* constant,
    const void* tolerance,
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
    const float tolerance2,
    int iterations
) {
    // 0.00111s
    const auto p = typename IterO::Logical(blockIdx, blockDim, threadIdx);
    if (iterO.isInBounds(p)) {
        const Complex<float> x = iterA[p];
        Complex<float> z = x;
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

cudaError_t srtMandelbrotFlat(
    srtDataType type,
    const void* pA,
    const void* pTolerance,
    size_t iterations,
    size_t count,
    void* pOut,
    cudaStream_t stream
) {
    assert(type == complex32F);
    const Complex<float>* a = static_cast<const Complex<float>*>(pA);
    float* out = static_cast<float*>(pOut);
    const float tolerance = *static_cast<const float*>(pTolerance);

    auto iterA = Flat(a, count);
    auto iterO = Flat(out, count);
    float tolerance2 = tolerance * tolerance;

    dim3 tile = tileSize(iterO.count);
    dim3 grid = gridSize(iterO.count, tile);

    mapMandelbrot<<<grid, tile, 0, stream>>>(iterA, iterO, tolerance2, iterations);
    return cudaSuccess;
}

