import XCTest
import SwiftRT

//------------------------------------------------------------------------------
// the actual tests
fileprivate extension ComputePlatform {
    func test_add() {
        let result = add(2, 3)
        XCTAssert(result == 5)
    }
    
    func test_addMore() {
        let result = addMore(2, 3)
        XCTAssert(result == 5)
    }
}

//------------------------------------------------------------------------------
// Platform variants
//#assert(test_AddTestsCpu.allTests.count == test_AddTestsCuda.allTests.count)

final class test_AddTestsCpu: XCTestCase {
    // type
    let platform = Platform<CpuService>()
    
    // static list
    static let allTests = [
        ("test_add", test_add),
    ]
    
    // delegates
    func test_add() { platform.test_add() }
    func test_addMore() { platform.test_addMore() }
}

final class test_AddTestsCuda: XCTestCase {
    // type
    let platform = Platform<CudaService>()
    
    // static list
    static let allTests = [
        ("test_add", test_add),
    ]
    
    // delegates
    func test_add() { platform.test_add() }
    func test_addMore() { platform.test_addMore() }
}

