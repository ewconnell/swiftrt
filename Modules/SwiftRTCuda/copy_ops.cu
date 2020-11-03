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
#include "copy_fn.cuh"
#include "op1.cuh"

//==============================================================================
// Swift importable C interface functions
//==============================================================================

//==============================================================================
// srtCopy
cudaError_t srtCopy(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream
) {
    // Cast2TensorDescriptorsA(paDesc, poDesc)
    return cudaErrorNotSupported;
}

cudaError_t srtCopyFlat(
    srtDataType atype,
    const void* a,
    srtDataType otype,
    void* out,
    size_t count,
    cudaStream_t stream
) {
    return select<CastOp>(atype, a, otype, out, count, stream);
}
