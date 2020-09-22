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
        // ("test_queueSync", test_queueSync),
        ("test_perfCurrentQueue", test_perfCurrentQueue),
        ("test_discreteMemoryReplication", test_discreteMemoryReplication),
        // ("test_multiQueueDependency", test_multiQueueDependency),
    ]

    // append and use a discrete async cpu device for these tests
    override func setUpWithError() throws {
//        log.level = .diagnostic
    }

    override func tearDownWithError() throws {
//        log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_queueSync() { testEachDevice(queueSync) }

    func queueSync() {
        let one = array([1, 1, 1, 1])
        let a = array([0, 1, 2, 3])
        let b = array([4, 5, 6, 7])

        // test app thread sync
        do {
            let c = a + b
            delayQueue(atLeast: 0.1)
            XCTAssert(c == [4, 6, 8, 10])
        }

        // test cross queue sync
        do {
            let c = a + b
            delayQueue(atLeast: 0.1)
            let d: Tensor1 = using(device: 0, queue: 1) {
                defer { delayQueue(atLeast: 0.1) }
                return c + one
            }
            let da = d.array
            XCTAssert(da == [5, 7, 9, 11])
        }
    }

    //--------------------------------------------------------------------------
    func test_perfCurrentQueue() {
        let _ = 0
        
        #if !DEBUG
        measure {
            for _ in 0..<1000000 {
                _ = currentQueue
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    func test_discreteMemoryReplication() {
        #if canImport(SwiftRTCuda)
        testEachDevice { discreteMemoryReplication() }
        #endif
    }
    
    func discreteMemoryReplication() {
        let a = array([[0, 1], [2, 3], [4, 5]], name: "a")
        let b = array([[0, 1], [2, 3], [4, 5]], name: "b")
        let c: Tensor2 = using(device: Platform.discreteMemoryDeviceId) {
            let result = a + b
            XCTAssert(a.storage.testLastAccessCopiedDeviceMemory)
            XCTAssert(b.storage.testLastAccessCopiedDeviceMemory)
            return result
        }
        let expected = c.array
        XCTAssert(c.storage.testLastAccessCopiedDeviceMemory)
        XCTAssert(expected == [[0, 2], [4, 6], [8, 10]])
    }
    
    //--------------------------------------------------------------------------
    func test_multiQueueDependency() {
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
        
        let da = d.array
        XCTAssert(da == [[0.0, 3.0], [6.0, 9.0], [12.0, 15.0]])
    }
}
