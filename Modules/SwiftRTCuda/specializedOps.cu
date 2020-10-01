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
#include "specializedOps.h"
#include "dispatchHelpers.h"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//------------------------------------------------------------------------------
// tensorA Element
template<typename IndexA, typename IndexO>
__global__ void mapJulia(
    const Complex<float>* __restrict__ a,
    IndexA indexA,
    float* __restrict__ out, 
    IndexO indexO,
    const float tolerance,
    const Complex<float> C,
    int iterations
) {
    // 0.003s
    const auto position = IndexO::Logical(blockIdx, blockDim, threadIdx);
    if (indexO.isInBounds(position)) {
        const int ia = indexA.linear(position);
        const int io = indexO.linear(position);

        auto t2 = tolerance * tolerance;
        auto Z = a[ia];
        auto d = out[io];
        for (int j = 0; j < iterations; ++j) {
            Z = Z * Z + C;
            auto m = min(d, float(j));
            d = abs2(Z) > t2 ? m : d;
        }
        out[io] = d;
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

    dim3 tile = tileSize(dDesc.count);
    dim3 grid = gridSize<1>(dDesc, tile);

    mapJulia<Flat,Flat><<<grid, tile, 0, stream>>>(
        z, zDesc,
        d, dDesc,
        tolerance, 
        C, 
        iterations);

    return cudaSuccess;
}
