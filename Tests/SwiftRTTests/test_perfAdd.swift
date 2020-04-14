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

class test_perfAdd: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_sumTensor2", test_sumTensor2),
        ("test_perfAddInApp", test_perfAddInApp),
        ("test_perfAddInModule", test_perfAddInModule),
    ]
    
    //--------------------------------------------------------------------------
    // test_sumTensor2
    func test_sumTensor2() {
        #if !DEBUG
        let a = ones((1024, 1024))
        var count: DType = 0
        
        // 0.001s
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
        #endif
    }

    //--------------------------------------------------------------------------
    // test_perfAddInApp
    func test_perfAddInApp() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = ones((1024, 1024))
        var count: DType = 0

        // THIS IS THE SAME CODE copied from DeviceQueue implementation
        func mapOp<T, U, R>(
            _ lhs: T, _ rhs: U, _ r: inout R,
            _ op: @escaping (T.Element, U.Element) -> R.Element)
            where T: Collection, U: Collection, R: MutableCollection
        {
            zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = op($1.0, $1.1) }
        }
        
        // 0.0470s
        self.measure {
            var result = empty(like: a)
            mapOp(a, b, &result, +)
            // keep things from being optimized away
            count += result.first
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_perfAddInModule
    // 30X slower than test_perfAddInApp!!
    func test_perfAddInModule() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = ones((1024, 1024))
        var count: DType = 0
        
        // 1.678s
        self.measure {
            let result = a + b
            count += result.first
        }
        XCTAssert(count > 0)
        #endif
    }
}
