configure_file(include/module.modulemap.in module.modulemap @ONLY)

# NOTE: we ignore shim.c as that is specifically for the benefit of Swift
# Package Manager.
add_library(CCUDA INTERFACE)
target_include_directories(CCUDA INTERFACE
  ${CMAKE_CURRENT_BINARY_DIR})
target_link_directories(CCUDA INTERFACE
  ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(CCUDA INTERFACE
  CUDA::cublasLt
  CUDA::cudart
  CUDA::cudnn
  CUDA::curand)
