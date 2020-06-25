//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import XCTest
import Foundation
import SwiftRT

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_add", test_add),
    ]
    
    // append and use a discrete async cpu device for these tests
    override func setUpWithError() throws {
        Context.log.level = .diagnostic
        Context.queuesPerDevice = 1
        use(device: 0, queue: 0)
    }

    override func tearDownWithError() throws {
        useSyncQueue()
        Context.log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_add() {
        let a = array([[0, 1], [2, 3], [4, 5]])
        let b = array([[0, 1], [2, 3], [4, 5]])
        let result = a + b
        XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
        
        // both
        let (g1, g2) = pullback(at: a, b, in: { $0 + $1 })(ones(like: a))
        
        XCTAssert(g1.flatArray == [1, 1, 1, 1, 1, 1])
        XCTAssert(g2.flatArray == [1, 1, 1, 1, 1, 1])
        
        // lhs
        let glhs = pullback(at: a, in: { $0 + 2 })(ones(like: a))
        XCTAssert(glhs.flatArray == [1, 1, 1, 1, 1, 1])
        
        // rhs
        let grhs = pullback(at: a, in: { 2 + $0 })(ones(like: a))
        XCTAssert(grhs.flatArray == [1, 1, 1, 1, 1, 1])
    }
}
