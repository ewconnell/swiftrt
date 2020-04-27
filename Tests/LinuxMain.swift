import XCTest
import SwiftRTCoreTests
import SwiftRTLayerTests

var tests = [XCTestCaseEntry]()
tests += SwiftRTCoreTests.allTests()
tests += SwiftRTLayerTests.allTests()
XCTMain(tests)
