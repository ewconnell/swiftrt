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
        ("test_addTensor2", test_addTensor2),
    ]
    
    //--------------------------------------------------------------------------
    // test_sumTensor2
    func test_sumTensor2() {
        #if !DEBUG
        let a = ones((1024, 1024))
        var count: DType = 0
        
        // 0.00116s
        self.measure {
            // llvm.experimental.vector.reduce.fadd
            count += a.indices.reduce(into: 0) { $0 += a[$1] }
        }

        XCTAssert(count > 0)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_addTensor2
    // 30X slower than test_perfAddInApp!!
    func test_addTensor2() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = ones((1024, 1024))
        var count: DType = 0
        
        // 1.678 --> 0.00412  ~400X improvement
        self.measure {
            count = (a + b).first
        }
        XCTAssert(count == 2)
        #endif
    }

    //--------------------------------------------------------------------------
    // test_perfAddScalarInApp
    func test_perfAddScalarInApp() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = repeating(1, (1024, 1024))
        var count: DType = 0
        
        func mapOp1<S,E>(
            _ lhs: Tensor<S,E>, _ rhs: E,
            _ r: inout Tensor<S,E>, _ op: (E, E) -> E)
        where S: TensorShape, E: AdditiveArithmetic
        {
            zip(r.indices, lhs).forEach { r[$0] = $1 + rhs }
        }

        func mapOp2<S,E>(
            _ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
            _ r: inout Tensor<S,E>, _ op: (E, E) -> E)
        where S: TensorShape, E: AdditiveArithmetic
        {
            zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = $1.0 + $1.1 }
        }
        
        // 0.0470 --> 0.00409
        self.measure {
            var result = empty(like: a)
//            if b.isSingleElement {
//                mapOp1(a, b[b.startIndex], &result, +)
//            } else {
                mapOp2(a, b, &result, +)
//            }

            // keep things from being optimized away
            count += result.first
        }
        XCTAssert(count > 0)
        #endif
    }
}
