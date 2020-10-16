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
#include "srttypes.cuh"


#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
//  srtTensorDescriptor
// make visible to Swift as C API
typedef struct {
    /// the TensorElement cuda data type
    srtDataType type;
    /// the number of dimensions
    uint32_t rank;
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

