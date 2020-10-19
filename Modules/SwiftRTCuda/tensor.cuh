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
#pragma once
#include <cublasLt.h>


#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
//  srtDataTypes
typedef enum {
    unknown     = -1,
    // floating point types
    real16F     = CUDA_R_16F,
    real16BF    = CUDA_R_16BF,
    real32F     = CUDA_R_32F,
    real64F     = CUDA_R_64F,
    complex16F  = CUDA_C_16F,
    complex16BF = CUDA_C_16BF,
    complex32F  = CUDA_C_32F,
    complex64F  = CUDA_C_64F,

    // integral types
    real1U,
    real4I      = CUDA_R_4I,
    real4U      = CUDA_R_4U,
    real8I      = CUDA_R_8I,
    real8U      = CUDA_R_8U, 
    real16I     = CUDA_R_16I,
    real16U     = CUDA_R_16U,
    real32I     = CUDA_R_32I,
    real32U     = CUDA_R_32U, 
    real64U     = CUDA_R_64U,
    real64I     = CUDA_R_64I,
    complex4I   = CUDA_C_4I, 
    complex4U   = CUDA_C_4U,
    complex8I   = CUDA_C_8I, 
    complex8U   = CUDA_C_8U,
    complex16I  = CUDA_C_16I,
    complex16U  = CUDA_C_16U,
    complex32I  = CUDA_C_32I,
    complex32U  = CUDA_C_32U,
    complex64I  = CUDA_C_64I,
    complex64U  = CUDA_C_64U,

    // bool types
    boolean
} srtDataType;

//==============================================================================
//  srtTensorDescriptor
// make visible to Swift as C API
typedef struct {
    /// the TensorElement cuda data type
    srtDataType type;
    /// the number of dimensions
    int32_t rank;
    /// the storage layout order
    cublasLtOrder_t order;
    /// the number of logical elements in the tensor
    size_t count;
    /// the number of physical storage elements spanned by the tensor
    size_t spanCount;
    /// the size of each dimension in the tensor
    const size_t* shape;
    /// the stride to the next storage element in the tensor for each dimension 
    const size_t* strides;
    /// the stride to the next logical element position for each dimension 
    const size_t* logicalStrides;
} srtTensorDescriptor;

#ifdef __cplusplus
}
#endif

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

// statically cast types from C interface to c++ type
#define Cast2TensorDescriptorsA(pa, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po);

#define Cast2TensorDescriptorsAB(pa, pb, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po);

#define Cast2TensorDescriptorsABC(pa, pb, pc, po) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& oDesc = static_cast<const TensorDescriptor&>(*po);

#define Cast2TensorDescriptorsABCOO(pa, pb, pc, po0, po1) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& bDesc = static_cast<const TensorDescriptor&>(*pb); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& o0Desc = static_cast<const TensorDescriptor&>(*po0); \
const TensorDescriptor& o1Desc = static_cast<const TensorDescriptor&>(*po1);

#define Cast2TensorDescriptorsAECOO(pa, pc, po0, po1) \
const TensorDescriptor& aDesc = static_cast<const TensorDescriptor&>(*pa); \
const TensorDescriptor& cDesc = static_cast<const TensorDescriptor&>(*pc); \
const TensorDescriptor& o0Desc = static_cast<const TensorDescriptor&>(*po0); \
const TensorDescriptor& o1Desc = static_cast<const TensorDescriptor&>(*po1);

