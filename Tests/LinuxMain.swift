import XCTest
import SwiftRTCoreTests
import SwiftRTLayerTests
import BenchmarkTests

var tests = [XCTestCaseEntry]()
tests += BenchmarkTests.allTests()
// tests += SwiftRTLayerTests.allTests()
// tests += SwiftRTCoreTests.allTests()
XCTMain(tests)
