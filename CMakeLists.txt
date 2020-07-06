cmake_minimum_required(VERSION 3.5)
project(swiftrt)

#####################
# output directories
string(TOLOWER "${CMAKE_BUILD_TYPE}" SWIFT_BUILD_TYPE)
set(TARGET_DIR ${CMAKE_SOURCE_DIR}/.build/${SWIFT_BUILD_TYPE})

#####################
# C module map directory for Swift import
set(MODULES_DIR ${PROJECT_SOURCE_DIR}/Modules)

file(GLOB_RECURSE MODULES_HEADERS
        ${MODULES_DIR}/*.h
        ${MODULES_DIR}/*.modulemap)

#####################
# Cuda
cmake_policy(SET CMP0074 NEW)
find_package(CUDA REQUIRED)

#####################
# CudaKernels static library
set(CUDA_KERNELS_LIB_NAME CudaKernels)
set(CUDA_KERNELS_DIR ${MODULES_DIR}/${CUDA_KERNELS_LIB_NAME})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Wno-deprecated-gpu-targets
        #        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        --compiler-options -fPIC)

if (${TARGET_DIR} MATCHES "debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG)
endif()

file(GLOB_RECURSE CUDA_KERNELS_SOURCES
        ${CUDA_KERNELS_DIR}/*.cu
        ${CUDA_KERNELS_DIR}/*.c
        ${CUDA_KERNELS_DIR}/*.h)

cuda_add_library(${CUDA_KERNELS_LIB_NAME} ${CUDA_KERNELS_SOURCES})

set_target_properties(${CUDA_KERNELS_LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${TARGET_DIR})

add_custom_command(TARGET ${CUDA_KERNELS_LIB_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR})

#####################
# swift
#   set SWIFT_HOME in /etc/environment for CLion to pick up
# then re-login to make visible after setting
IF(DEFINED ENV{SWIFT_HOME})
    set(SWIFT_HOME $ENV{SWIFT_HOME})
ELSE()
    set(SWIFT_HOME /usr/local/swift)
ENDIF()

set(SWIFT ${SWIFT_HOME}/usr/bin/swift)
set(SWIFTC ${SWIFT_HOME}/usr/bin/swiftc)
# set(SWIFT_CLANG_INCLUDE ${SWIFT_HOME}/usr/lib/swift/clang/include)
#        -Xswiftc -I${SWIFT_CLANG_INCLUDE}

# configuration specific swift flags
set(SWIFT_CONFIG_FLAGS
        # -Xswiftc -warnings-as-errors
        # -Xswiftc -j12
        -Xswiftc -I/usr/local/include
        -Xswiftc -L/usr/local/lib
        -Xswiftc -I${MODULES_DIR}/CudaKernels/include
        -Xswiftc -I${MODULES_DIR}/Cuda/include
        -Xswiftc -I${MODULES_DIR}
        -Xswiftc -L${TARGET_DIR}
        -Xlinker -l${CUDA_KERNELS_LIB_NAME}
        -Xswiftc -I${CUDA_INCLUDE_DIRS}
        -Xlinker -L${CUDA_TOOLKIT_ROOT_DIR}/lib64
        )

#####################
# SwiftRT module
set(SWIFTRT_NAME SwiftRT)
file(GLOB_RECURSE SWIFTRT_SOURCES  ${PROJECT_SOURCE_DIR}/Sources/*.swift)

set(SWIFTRT_FLAGS ${SWIFT_CONFIG_FLAGS})

add_custom_target(${SWIFTRT_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${SWIFTRT_FLAGS}
        SOURCES ${SWIFTRT_SOURCES} ${MODULES_HEADERS})

add_dependencies(${SWIFTRT_NAME} ${CUDA_KERNELS_LIB_NAME})

#####################
# SwiftRTTests app
set(SWIFTRT_TESTS_NAME SwiftRTTests)
file(GLOB_RECURSE SWIFTRT_TEST_SOURCES
        ${PROJECT_SOURCE_DIR}/Tests/${SWIFTRT_TESTS_NAME}/*.swift)

#set(TEST_CASE -s ${SWIFTRT_TESTS_NAME}.TestSetup/test_mnistForward)

add_custom_target(${SWIFTRT_TESTS_NAME}
        COMMAND ${SWIFT} build --build-tests ${SWIFTRT_FLAGS}
        SOURCES ${SWIFTRT_TEST_SOURCES})

add_dependencies(${SWIFTRT_TESTS_NAME} ${SWIFTRT_NAME} ${CUDA_KERNELS_LIB_NAME})
