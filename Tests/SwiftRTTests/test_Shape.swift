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

class test_Shape: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_SequentialViews", test_SequentialViews),
        ("test_transposed", test_transposed),
    ]

    //--------------------------------------------------------------------------
    // test_perfIndexTensor1
    func test_perfIndexTensor1() {
//        #if !DEBUG
        let a = ones(1024 * 1024)
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
//        #endif
    }

    func test_initRepeating() {
        //        #if !DEBUG
        var count: DType = 0
        self.measure {
            for _ in 0..<100000 {
                let a = Tensor1<Float>(repeating: 1, to: Shape1(1))
                count += a.element
            }
        }
        XCTAssert(count > 0)
        //        #endif
    }
    
    func test_initSingle() {
        //        #if !DEBUG
        var count: DType = 0
        self.measure {
            for _ in 0..<100000 {
                let a = Tensor1<Float>(1)
                count += a.element
            }
        }
        XCTAssert(count > 0)
        //        #endif
    }

    func test_perfRepeatedTensor3() {
        //        #if !DEBUG
        let a = repeating(1, (64, 128, 128))
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
        //        #endif
    }

    func test_perfIndexShape2() {
//        #if !DEBUG
        let a = ones((1024, 1024))
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
//        #endif
    }

    func test_perfIndexShape3() {
//        #if !DEBUG
        let a = ones((64, 128, 128))
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
//        #endif
    }
    
    func test_perfIndexShape4() {
//        #if !DEBUG
        let a = ones((2, 32, 128, 128))
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
//        #endif
    }
    
    func test_perfIndexShape5() {
//        #if !DEBUG
        let a = ones((2, 2, 16, 128, 128))
        var count: DType = 0
        self.measure {
            for value in a {
                count += value
            }
        }
        XCTAssert(count > 0)
//        #endif
    }

    //--------------------------------------------------------------------------
    // test_SequentialViews
    let simdPerfIterations = 10000000
    
    func test_perfSIMD2() {
//        #if !DEBUG
        let bounds = SIMD2(2, 3)
        let strides = SIMD2(3, 1)
        let pos = SIMD2(1, 2)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() &+ 1
                let count = bounds.indices.reduce(1) { $0 &* bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span &+ count &+ linear
            }
        }
        XCTAssert(total > 0)
//        #endif
    }

    func test_perfSIMD3() {
//        #if !DEBUG
        let bounds = SIMD3(2, 3, 4)
        let strides = SIMD3(12, 3, 1)
        let pos = SIMD3(1, 2, 3)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
//        #endif
    }
    
    func test_perfSIMD4() {
//        #if !DEBUG
        let bounds = SIMD4(1, 2, 3, 4)
        let strides = SIMD4(24, 12, 3, 1)
        let pos = SIMD4(0, 1, 2, 3)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
//        #endif
    }

    func test_perfSIMD5() {
//        #if !DEBUG
        let bounds = SIMD5(1, 2, 3, 4, 5)
        let strides = SIMD5(120, 60, 20, 5, 1)
        let pos = SIMD5(0, 1, 2, 3, 4)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
//        #endif
    }
    //--------------------------------------------------------------------------
    // test_SequentialViews
    func test_SequentialViews() {
        // vector views are always sequential
        let v = array(0..<6)
        let subv = v[1...2]
        XCTAssert(subv.isSequential)
        
        // a batch of rows are sequential
        let m = empty((4, 5))
        let mrows = m[1...2, ...]
        XCTAssert(mrows.isSequential)

        // a batch of columns are not sequential
        let m1 = empty((4, 5))
        let mcols = m1[..., 1...2]
        XCTAssert(!mcols.isSequential)
    }
    
    //--------------------------------------------------------------------------
    // test_transposed
    func test_transposed() {
        let m = array(0..<9, (3, 3))
        XCTAssert(m.t == [[0,3,6], [1,4,7], [2,5,8]])

        let a = array(0..<24, (2,3,4))
        let transA = a.transposed(permutatedBy: (2,1,0))
        XCTAssert(transA == [[[ 0.0, 12.0],
                              [ 4.0, 16.0],
                              [ 8.0, 20.0]],

                             [[ 1.0, 13.0],
                              [ 5.0, 17.0],
                              [ 9.0, 21.0]],

                             [[ 2.0, 14.0],
                              [ 6.0, 18.0],
                              [10.0, 22.0]],

                             [[ 3.0, 15.0],
                              [ 7.0, 19.0],
                              [11.0, 23.0]]])
    }
}
