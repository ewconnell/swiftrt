add_library(SwiftRTCuda STATIC
  math_ops.cu)
set_target_properties(SwiftRTCuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION YES
  POSITION_INDEPENDENT_CODE YES)
target_compile_options(SwiftRTCuda PRIVATE
  -allow-unsupported-compiler
  --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets)
target_include_directories(SwiftRTCuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_precompile_headers(SwiftRTCuda PRIVATE precomp.hpp)
