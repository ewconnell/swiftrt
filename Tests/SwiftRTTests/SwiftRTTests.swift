import XCTest
import SwiftRT

//------------------------------------------------------------------------------
// the actual tests
fileprivate extension Platform {
    func test_add() {
        let v0 = Vector(with: [2])
        let v1 = Vector(with: [3])
        let result = add(v0, v1)
        XCTAssert(result.element == 5)
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
}

final class test_AddTestsCuda: XCTestCase {
    // type
    let platform = Platform<CpuService>()
    
    // static list
    static let allTests = [
        ("test_add", test_add),
    ]
    
    // delegates
    func test_add() { platform.test_add() }
}

