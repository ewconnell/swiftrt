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
#include "disp.h"
#include "mathOps.h"
#include "mathSupplemental.h"
#include <type_traits>


//==============================================================================
// Swift importable C interface functions
//==============================================================================

IntFloatComplexOpA(Abs, abs)

cudaError_t srtAbs(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectA<Abs>(a, aDesc, out, oDesc, stream);
}

FloatOpA(Acos, acos)

// Must be promoted types
cudaError_t srtAcos(
    const void* a, const srtTensorDescriptor* paDesc,
    void* out, const srtTensorDescriptor* poDesc,
    cudaStream_t stream)
{
    Cast2TensorDescriptorsA(paDesc, poDesc)
    return selectA<Acos>(a, aDesc, out, oDesc, stream);
}
