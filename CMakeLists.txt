cmake_minimum_required(VERSION 3.18)
project(swiftrt
  LANGUAGES Swift)
include(CTest)

option(SWIFTRT_ENABLE_CUDA "Enable CUDA Support" NO)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

<<<<<<< HEAD
#####################
# C module map directory for Swift import
set(MODULES_DIR ${PROJECT_SOURCE_DIR}/Modules)

file(GLOB_RECURSE MODULES_HEADERS
        ${MODULES_DIR}/*.h
        ${MODULES_DIR}/*.modulemap)

#####################
# Cuda
if ($ENV{SWIFTRT_PLATFORM} MATCHES "cuda")
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
                --compiler-options -fPIC
                #-gencode arch=compute_80,code=sm_80
                -gencode arch=compute_75,code=sm_75
                #-gencode arch=compute_70,code=sm_70
                #-gencode arch=compute_60,code=sm_60
                #-gencode arch=compute_53,code=sm_53
                )

        if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG -g -G)
        else()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -Xptxas -O3)
        endif()

        file(GLOB_RECURSE SWIFTRT_CUDA_SOURCES
                ${SWIFTRT_CUDA_DIR}/*.cu
                ${SWIFTRT_CUDA_DIR}/*.c
                ${SWIFTRT_CUDA_DIR}/*.cpp
                ${SWIFTRT_CUDA_DIR}/*.h
                ${SWIFTRT_CUDA_DIR}/*.cuh)

        cuda_add_library(${SWIFTRT_CUDA_LIB_NAME} ${SWIFTRT_CUDA_SOURCES})
=======
if(SWIFTRT_ENABLE_CUDA)
  enable_language(CUDA)
  add_subdirectory(Modules/SwiftRTCuda)
>>>>>>> origin/master
endif()

# External Projects
include(ExternalProject)

function(import_module module_name build_dir build_target)
  add_library(${module_name} IMPORTED UNKNOWN)
  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set_target_properties(${module_name} PROPERTIES
      IMPORTED_IMPLIB ${build_dir}/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}${module_name}${CMAKE_IMPORT_LIBRARY_SUFFIX}
      INTERFACE_INCLUDE_DIRECTORIES ${build_dir}/swift)
  else()
    set_target_properties(${module_name} PROPERTIES
      IMPORTED_LOCATION ${build_dir}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}${module_name}${CMAKE_SHARED_LIBRARY_SUFFIX}
      INTERFACE_INCLUDE_DIRECTORIES ${build_dir}/swift)
  endif()
  add_dependencies(${module_name} ${build_target})
endfunction()

## swift-numerics
ExternalProject_Add(swift-numerics
  GIT_REPOSITORY
    git://github.com/apple/swift-numerics
  GIT_TAG
    master
  CMAKE_ARGS
    -D BUILD_SHARED_LIBS=YES
    -D BUILD_TESTING=NO
    -D CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -D CMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -D CMAKE_Swift_COMPILER_TARGET=${CMAKE_Swift_COMPILER_TARGET}
    -D CMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
  INSTALL_COMMAND
    ""
  BUILD_BYPRODUCTS
    <BINARY_DIR>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}Numerics${CMAKE_SHARED_LIBRARY_SUFFIX}
    <BINARY_DIR>/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}Numerics${CMAKE_IMPORT_LIBRARY_SUFFIX}
  UPDATE_DISCONNECTED
    TRUE
  STEP_TARGETS
    build)
ExternalProject_Get_Property(swift-numerics BINARY_DIR)
ExternalProject_Get_Property(swift-numerics SOURCE_DIR)

file(MAKE_DIRECTORY ${BINARY_DIR}/swift)
file(MAKE_DIRECTORY ${SOURCE_DIR}/Sources/_NumericsShims/include)

import_module(Numerics ${BINARY_DIR} swift-numerics-build)
import_module(ComplexModule ${BINARY_DIR} swift-numerics-build)
import_module(RealModule ${BINARY_DIR} swift-numerics-build)

add_library(_NumericsShims IMPORTED INTERFACE)
set_target_properties(_NumericsShims PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR}/Sources/_NumericsShims/include)
add_dependencies(_NumericsShims swift-numerics-build)

set_target_properties(Numerics PROPERTIES
  INTERFACE_LINK_DIRECTORIES ${BINARY_DIR}/lib
  INTERFACE_LINK_LIBRARIES _NumericsShims)

find_package(dispatch CONFIG QUIET)
find_package(Foundation CONFIG QUIET)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_compile_definitions(DEBUG)
endif()

add_subdirectory(Sources)
if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()
