cmake_minimum_required(VERSION 3.18)
project(swiftrt
  LANGUAGES Swift)
include(CTest)

option(SWIFTRT_ENABLE_CUDA "Enable CUDA Support" NO)
option(SWIFTRT_LOGLEVEL_DIAGNOSTIC "Set the default Log level to diagnostic" NO)
option(SWIFTRT_RELEASE_UNCHECKED "Enable Unchecked Release build" NO)

#--------------------------------------
# Report build configuration
if((CMAKE_BUILD_TYPE STREQUAL Release) AND SWIFTRT_RELEASE_UNCHECKED)
  message(STATUS "SwiftRT build type is Release Unchecked")
  add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:Swift>:-Ounchecked -enforce-exclusivity=unchecked>")
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
  add_compile_definitions(DEBUG)
  message(STATUS "SwiftRT build type is Debug")
else()
  message(STATUS "SwiftRT build type is ${CMAKE_BUILD_TYPE}")
endif()
#--------------------------------------

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

if(SWIFTRT_ENABLE_CUDA)
  if(CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "SwiftRT cuda support is enabled for SM architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CXX)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(CUDAToolkit_FOUND)
      _CUDAToolkit_find_and_add_import_lib(cudnn)
    endif()

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED YES)

    add_subdirectory(Sources/CCUDA)
    add_subdirectory(Modules/SwiftRTCuda)
  else()
    message(FATAL_ERROR
      "When CUDA is enabled, CMAKE_CUDA_ARCHITECTURES must be specified.\n"
      "Example: -DCMAKE_CUDA_ARCHITECTURES=\"60;75\"\n")
  endif()
else()
  message(STATUS "SwiftRT cuda support is disabled.")
endif()

if(SWIFTRT_LOGLEVEL_DIAGNOSTIC)
  add_compile_definitions(LOGLEVEL_DIAGNOSTIC)
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
  GIT_REPOSITORY git://github.com/apple/swift-numerics
  GIT_TAG origin/master
  CMAKE_ARGS
    -D BUILD_SHARED_LIBS=YES
    -D BUILD_TESTING=NO
    -D CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    -D CMAKE_Swift_COMPILER=${CMAKE_Swift_COMPILER}
    -D CMAKE_Swift_COMPILER_TARGET=${CMAKE_Swift_COMPILER_TARGET}
    -D CMAKE_Swift_FLAGS=${CMAKE_Swift_FLAGS}
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
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

add_subdirectory(Sources)
if(BUILD_TESTING)
  add_subdirectory(Tests)
endif()