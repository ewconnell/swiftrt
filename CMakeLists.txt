cmake_minimum_required(VERSION 3.18)
project(swiftrt
  LANGUAGES Swift)
include(CTest)

option(SWIFTRT_ENABLE_CUDA "Enable CUDA Support" NO)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)

if(SWIFTRT_ENABLE_CUDA)
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED YES)

  add_subdirectory(Modules/SwiftRTCuda)
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
