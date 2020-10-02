add_library(SwiftRTCuda STATIC
  delayStream.cu
  fillOps.cu
  mathOps.cu
  randomOps.cu)
target_compile_options(SwiftRTCuda PRIVATE
  -allow-unsupported-compiler
  --std c++17
  --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets
  -gencode arch=compute_75,code=sm_75
  --compiler-options -fPIC)
target_include_directories(SwiftRTCuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
