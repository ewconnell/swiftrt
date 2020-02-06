import XCTest
import SwiftRT

// override func setUp() {
// override func tearDown() {

final class SwiftRTCpuTests: XCTestCase {
    static var allTests = [
        ("test_add", test_add),
    ]

    let platform = Platform<CpuService>()

    func test_add() { platform.test_add() }
    func test_addMore() { platform.test_addMore() }
}

final class SwiftRTCudaTests: XCTestCase {
    static var allTests = [
        ("test_add", test_add),
    ]
    
    let platform = Platform<CudaService>()
    
    func test_add() { platform.test_add() }
    func test_addMore() { platform.test_addMore() }
}

extension Platform {
    func test_add() {
        let result = add(2, 3)
        XCTAssert(result == 5)
    }
    
    func test_addMore() {
        let result = addMore(2, 3)
        XCTAssert(result == 5)
    }
}
