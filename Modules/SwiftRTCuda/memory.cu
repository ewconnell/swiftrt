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
#include "memory_api.h"

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
using namespace cub;

#include <stdio.h>

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory



cudaError_t srtDeviceAllocate(void **devPtr, size_t size, cudaStream_t stream) {
    // printf("srtDeviceAllocate: %ld\n", size);
    return g_allocator.DeviceAllocate(devPtr, size, stream);
}

cudaError_t srtDeviceFree(void *devPtr) {
    return g_allocator.DeviceFree(devPtr);
}
