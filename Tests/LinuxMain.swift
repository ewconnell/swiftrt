import XCTest
import SwiftRTCoreTests
import SwiftRTLayerTests

var tests = [XCTestCaseEntry]()
// tests += SwiftRTLayerTests.allTests()
tests += SwiftRTCoreTests.allTests()
XCTMain(tests)
