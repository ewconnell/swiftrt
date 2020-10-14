
find_package(XCTest CONFIG QUIET)

add_subdirectory(BenchmarkTests)
add_subdirectory(SwiftRTCoreTests)
add_subdirectory(SwiftRTLayerTests)

add_executable(SwiftRTTestRunner
  LinuxMain.swift)
target_link_libraries(SwiftRTTestRunner PRIVATE
  BenchmarkTests
  SwiftRTCoreTests
  SwiftRTLayerTests)

add_test(NAME SwiftRTTests
  COMMAND SwiftRTTestRunner)
