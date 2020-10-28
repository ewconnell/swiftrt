add_library(SwiftRTCuda
  compare_ops.cu
  utilities.cu
  fill_ops.cu
  math_ops.cu
  memory.cu
  random_ops.cu
  reduce_ops.cu
  specialized_ops.cu
)
set_target_properties(SwiftRTCuda PROPERTIES
  #CUDA_SEPARABLE_COMPILATION YES
  POSITION_INDEPENDENT_CODE YES)
target_compile_options(SwiftRTCuda PRIVATE
  -allow-unsupported-compiler
  --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets)
target_include_directories(SwiftRTCuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SwiftRTCuda PUBLIC
  CCUDA)
#target_precompile_headers(SwiftRTCuda PRIVATE precomp.hpp)
