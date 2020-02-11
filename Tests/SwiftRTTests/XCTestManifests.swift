import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(test_BinaryFunctions.allTests),
    ]
}
#endif
