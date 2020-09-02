cmake_minimum_required(VERSION 3.5)

# This must come before the `project` declaration
# we specify less than clang 10 because nvcc doesn't support v10 yet
# Hmm declaring this AND setting the Vscode kit to clang 9 causes an
# infinite loop in the cmake generator.
#
#set(CMAKE_C_COMPILER "/usr/lib/llvm-9/bin/clang")
#set(CMAKE_CXX_COMPILER "/usr/lib/llvm-9/bin/clang")

project(swiftrt)

#####################
# output directories
string(TOLOWER "${CMAKE_BUILD_TYPE}" SWIFT_BUILD_TYPE)
set(TARGET_DIR ${PROJECT_SOURCE_DIR}/.build/${SWIFT_BUILD_TYPE})

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
# SwiftRTCuda static library
set(SWIFTRT_CUDA_LIB_NAME SwiftRTCuda)
set(SWIFTRT_CUDA_DIR ${MODULES_DIR}/${SWIFTRT_CUDA_LIB_NAME})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -allow-unsupported-compiler
        --std c++17
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
        -gencode arch=compute_75,code=sm_75
        --compiler-options -fPIC)

if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG -g -G)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xptxas -O3)
endif()

file(GLOB_RECURSE SWIFTRT_CUDA_SOURCES
        ${SWIFTRT_CUDA_DIR}/*.cu
        ${SWIFTRT_CUDA_DIR}/*.c
        ${SWIFTRT_CUDA_DIR}/*.cpp
        ${SWIFTRT_CUDA_DIR}/*.h)

cuda_add_library(${SWIFTRT_CUDA_LIB_NAME} ${SWIFTRT_CUDA_SOURCES})

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
        -Xswiftc -I${MODULES_DIR}
        -Xswiftc -I${SWIFTRT_CUDA_DIR}
        -Xlinker -l${SWIFTRT_CUDA_LIB_NAME}
        -Xswiftc -L${CMAKE_BINARY_DIR}
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

add_dependencies(${SWIFTRT_NAME} ${SWIFTRT_CUDA_LIB_NAME})

#####################
# SwiftRTTests app
set(SWIFTRT_TESTS_NAME SwiftRTTests)
file(GLOB_RECURSE SWIFTRT_TEST_SOURCES
        ${PROJECT_SOURCE_DIR}/Tests/${SWIFTRT_TESTS_NAME}/*.swift)

#set(TEST_CASE -s ${SWIFTRT_TESTS_NAME}.TestSetup/test_mnistForward)

add_custom_target(${SWIFTRT_TESTS_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${SWIFTRT_FLAGS} --build-tests 
        SOURCES ${SWIFTRT_TEST_SOURCES} ${MODULES_HEADERS})

add_dependencies(${SWIFTRT_TESTS_NAME} ${SWIFTRT_NAME} ${SWIFTRT_CUDA_LIB_NAME})

# whenever the custom cuda kernel libraries are build, delete the exe
# to force swift build to relink
add_custom_command(TARGET ${SWIFTRT_CUDA_LIB_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove ${TARGET_DIR}/SwiftRTPackageTests.xctest)
