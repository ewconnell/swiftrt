add_library(SwiftRTCuda STATIC
  math_ops.cu
)
target_compile_options(SwiftRTCuda PRIVATE
  -allow-unsupported-compiler
  --std c++17
  --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets
  --compiler-options -fPIC)
target_include_directories(SwiftRTCuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_precompile_headers(SwiftRTCuda PRIVATE precomp.cuh)