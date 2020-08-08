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
#ifndef commonCDefs_h
#define commonCDefs_h

#include <stdint.h>
#include <cuda_runtime.h>


// make visible to Swift as C API
#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// srtTensorDescriptor
typedef struct {
    /// the number of dimensions
    uint32_t rank;
    /// the TensorElement cuda data type
    cudaDataType_t type;
    /// the number of logical elements in the tensor
    size_t count;
    /// the number of physical storage elements spanned by the tensor
    size_t spanCount;
    /// the size of each dimension in the tensor
    const size_t* shape;
    /// the stride to the next storage element in the tensor for each dimension 
    const size_t* strides;
} srtTensorDescriptor;

//==============================================================================
#ifdef __cplusplus
}
#endif

#endif // commonCDefs_h