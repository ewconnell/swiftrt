cmake_minimum_required(VERSION 3.5)
project(SwiftRTCudaKernels)

#####################
# output directories
string(TOLOWER "${CMAKE_BUILD_TYPE}" SWIFT_BUILD_TYPE)
set(TARGET_DIR ${CMAKE_SOURCE_DIR}/.build/${SWIFT_BUILD_TYPE})
set(PLUGINS_DIR ${TARGET_DIR}/PlugIns)
find_package(CUDA QUIET REQUIRED)

#####################
# CudaKernels lib
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Wno-deprecated-gpu-targets
#        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61)
#        --compiler-options -fPIC)

# for emulator debug -G -g -deviceemu
# for cuda-memcheck  -Xcompiler -rdynamic -lineinfo -G
if (${TARGET_DIR} MATCHES "debug")
    # adding the debug flags seems to make the swift debugger crazy
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG)
endif()

# CudaKernels static library
set(CUDA_KERNELS_NAME SwiftRTCudaKernels)
file(GLOB_RECURSE CUDA_KERNELS_SOURCES *.cu *.c *.h)
cuda_add_library(${CUDA_KERNELS_NAME} ${CUDA_KERNELS_SOURCES})

set_target_properties(${CUDA_KERNELS_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${TARGET_DIR})

add_custom_command(TARGET ${CUDA_KERNELS_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR})
