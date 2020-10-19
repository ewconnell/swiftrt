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
#include "specialized_api.cuh"
#include "complex.cuh"
#include "iterators.cuh"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
// tensorA Element
template<typename IterA, typename IterO>
__global__ void mapJulia(
    IterA iterA,
    IterO iterO,
    const float tolerance,
    const Complex<float> C,
    int iterations
) {
    // 0.000790s
    const auto p = IterO::Logical(blockIdx, blockDim, threadIdx);
    if (iterO.isInBounds(p)) {
        float t2 = tolerance * tolerance;
        Complex<float> Z = iterA[p];
        auto index = iterO.linear(p);
        float d = iterO[index];
        for (int j = 0; j < iterations; ++j) {
            Z = Z * Z + C;
            if (abs2(Z) > t2) {
                d = min(d, float(j));
                break;
            }
        }
        iterO[index] = d;
    }
}


cudaError_t srtJulia(
    const void* pz, const srtTensorDescriptor* pzDesc,
    void* pdivergence, const srtTensorDescriptor* pdDesc,
    const void* ptolerance,
    const void* pC,
    size_t iterations,
    cudaStream_t stream
) {
    const TensorDescriptor& zDesc = static_cast<const TensorDescriptor&>(*pzDesc);
    const TensorDescriptor& dDesc = static_cast<const TensorDescriptor&>(*pdDesc);

    const Complex<float>* z = static_cast<const Complex<float>*>(pz);
    float* d = static_cast<float*>(pdivergence);
    const float tolerance = *static_cast<const float*>(ptolerance);
    const Complex<float> C = *static_cast<const Complex<float>*>(pC);

    auto iterZ = Flat(z, zDesc.count);
    auto iterD = Flat(d, dDesc.count);

    dim3 tile = tileSize(iterD.count);
    dim3 grid = gridSize(iterD.count, tile);

    mapJulia<<<grid, tile, 0, stream>>>(iterZ, iterD, tolerance, C, iterations);
    return cudaSuccess;
}
