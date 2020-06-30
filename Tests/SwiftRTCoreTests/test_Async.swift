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
        ("test_multiQueueDependency", test_multiQueueDependency),
    ]
    
    // append and use a discrete async cpu device for these tests
    override func setUpWithError() throws {
//        Context.log.level = .diagnostic
        Context.cpuQueueCount = 2
        use(device: 0, queue: 0)
    }

    override func tearDownWithError() throws {
        useSyncQueue()
        Context.log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_multiQueueDependency() {
        do {
            let a = array([[0, 1], [2, 3], [4, 5]], name: "a")
            
            var c: Tensor2 = using(queue: 0) {
                let b = array([[0, 1], [2, 3], [4, 5]], name: "b")
                return a + b
            }
            c.name = "c"
            
            var d = using(queue: 1) {
                a + c
            }
            d.name = "d"
            
            let result = d.array
            XCTAssert(result == [[0.0, 3.0], [6.0, 9.0], [12.0, 15.0]])
        }
    }
}
