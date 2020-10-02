
add_library(SwiftRTLayerTests
  TestHelpers.swift
  XCTestManifests.swift
  test_Convolution.swift
  test_Dense.swift
  test_Recurrent.swift)
target_link_libraries(SwiftRTLayerTests PUBLIC
  $<$<AND:$<BOOL:Foundation_FOUND>,$<NOT:$<PLATFORM_ID:Darwin>>>:Foundation>
  $<$<BOOL:XCTest_Found>:XCTest>
  SwiftRT
  SwiftRTLayers)
