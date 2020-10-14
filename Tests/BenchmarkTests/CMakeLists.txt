
add_library(BenchmarkTests
  XCTestManifests.swift
  test_Fractals.swift)
target_link_libraries(BenchmarkTests PUBLIC
  $<$<AND:$<BOOL:Foundation_FOUND>,$<NOT:$<PLATFORM_ID:Darwin>>>:Foundation>
  $<$<BOOL:XCTest_Found>:XCTest>
  SwiftRT)
