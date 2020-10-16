add_library(SwiftRTCuda STATIC
  compare_ops.cu
  delayStream.cu
  fill_ops.cu
  math_ops.cu
  random_ops.cu
  reduce_ops.cu
  specialized_ops.cu)
target_compile_options(SwiftRTCuda PRIVATE
  -allow-unsupported-compiler
  --std c++17
  --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets
  --compiler-options -fPIC)
target_include_directories(SwiftRTCuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_precompile_headers(SwiftRTCuda PRIVATE precomp.cuh)