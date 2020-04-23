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

class test_Vectorizing: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_sumTensor2", test_reduceSumAll),
        ("test_addTensor2", test_AplusBSequential),
    ]
    
    //--------------------------------------------------------------------------
    // test_reduceSumAll
    func test_reduceSumAll() {
        #if !DEBUG
        let a = ones((1024, 1024))
        var count: DType = 0
        
        // 0.00140s
        self.measure {
//            for value in a {
//                count += value
//            }
            // llvm.experimental.vector.reduce.fadd
            count = a.indices.reduce(into: 0) { $0 += a[$1] }
        }

        XCTAssert(count > 0)
        #endif
    }

    //--------------------------------------------------------------------------
    // test_reduceMinAll
    func test_reduceMinAll() {
        #if !DEBUG
        let size = 1024
        let a = array(1...(size * size), (size, size))
        var value: DType = 0
        
        // 0.00192s
        self.measure {
            // llvm.experimental.vector.reduce.min
            value = a.indices.reduce(into: a[a.startIndex]) { $0 = Swift.min($0, a[$1]) }
        }
        
        XCTAssert(value == 1)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_AlessOrEqualBAny
    func test_AlessOrEqualBAny() {
        #if !DEBUG
        let size = 1024
        let a = array(1...(size * size), (size, size))
        let b = array(0..<(size * size), (size, size))
        var value = true
        
//        func reductionOp<S,E>(
//            _ x: Tensor<S,E>,
//            _ op: @escaping (E, E) -> E
//        ) -> E
//        {
//            // maps to llvm.experimental.vector.reduce.X
//            x.indices.reduce(into: x[x.startIndex]) { $0 = op($0, x[$1]) }
//        }
        func reductionOp<T>(_ x: T) -> T.Element
        where T: Collection, T.Element == Bool
        {
            // maps to llvm.experimental.vector.reduce.X
            let v = x.indices.reduce(into: x[x.startIndex]) { $0 = $0 && x[$1] }
            return v
        }

        // .00418s
        self.measure {
//            value = (a .<= b).any().element
            
            let x = a .<= b
//            value = x.indices.reduce(into: x[x.startIndex]) { $0 = $0 && x[$1] }
            value = reductionOp(x)

//            value = Context.currentQueue.reductionOp(comp, { $0 && $1 })
//            value = comp.indices.reduce(into: comp[comp.startIndex]) { $0 = $0 && comp[$1] }
        }
        
        XCTAssert(value == false)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_AplusBSequential
    func test_AplusBSequential() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = ones((1024, 1024))
        var count: DType = 0
        
        // 0.00412
        self.measure {
            let result = a + b
            count = result.first
        }
        XCTAssert(count > 0)
        #endif
    }

    //--------------------------------------------------------------------------
    // test_AplusBSequential
    func test_AplusB_NonSequential() {
        #if !DEBUG
        let size = 1024
        let a = array(1...(size * size), (size, size))
        let b = array(1...(size * size), (size, size), order: .F)
        var count: DType = 0
        
        // 0.00412
        self.measure {
            let result = a + b
            count = result.first
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_multiplyAdd
    func test_multiplyAdd() {
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
        where S: TensorShape, E: Numeric
        {
            zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = $1.0 * $1.1 + 1 }
        }
        
        // 0.00411
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
