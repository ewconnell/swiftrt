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

class test_Reductions: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_sumTensor3AlongAxes", test_sumTensor3AlongAxes),
        ("test_minTensor3AlongAxes", test_minTensor3AlongAxes),
        ("test_maxTensor3AlongAxes", test_maxTensor3AlongAxes),
        ("test_sumTensor1", test_sumTensor1),
        ("test_sumTensor2", test_sumTensor2),
        ("test_abssumTensor2", test_abssumTensor2),
        ("test_allTensor1", test_allTensor1),
        ("test_anyTensor1", test_anyTensor1),
        ("test_meanTensor2", test_meanTensor2),
        ("test_maxTensor2", test_maxTensor2),
        ("test_minTensor2", test_minTensor2),
        ("test_absmaxTensor2", test_absmaxTensor2),
        ("test_sqrtSumSquaresTensor2", test_sqrtSumSquaresTensor2),
        ("test_sqrtSumSquaresTensor3", test_sqrtSumSquaresTensor3),
    ]

    //--------------------------------------------------------------------------
    // test_gather
    // TODO: get this verified
    func test_gather() {
        let a = array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])
        let ai = array([0, 2], dtype: DeviceIndex.self)
        let b = gather(from: a, indices: ai)
        XCTAssert(b == [
            [0, 1, 2],
            [6, 7, 8]
        ])
        
        let c = gather(from: a, indices: ai, axis: 1)
        XCTAssert(c == [
            [0, 2],
            [3, 5],
            [6, 8]
        ])

        let g0 = gradient(at: ones(like: a)) {
            gather(from: $0 * a, indices: ai).sum().element
        }
        XCTAssert(g0 == [
            [0, 1, 2],
            [0, 0, 0],
            [6, 7, 8]
        ])

        let g1 = gradient(at: ones(like: a)) {
            gather(from: $0 * a, indices: ai, axis: -1).sum().element
        }
        XCTAssert(g1 == [
            [0, 0, 2],
            [3, 0, 5],
            [6, 0, 8]
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_sumTensor3AlongAxes
    func test_sumTensor3AlongAxes() {
//        Context.log.level = .diagnostic
        let v = array(
            [[[10,   2],
              [ 3,   4],
              [ 5,   6]],
             
             [[ 1,   2],
              [ 3,   4],
              [ 5,   6]]])

        // sum depths
        XCTAssert(v.sum(alongAxes: 0) == [
            [
                [11,  4],
                [ 6,  8],
                [10, 12]
            ]
        ])
        
        // sum rows
        XCTAssert(v.sum(alongAxes: 1) == [
            [
                [18, 12]
            ],
            [
                [ 9, 12]
            ]
        ])

        // sum columns
        XCTAssert(v.sum(alongAxes: 2) == [
            [
                [12],
                [ 7],
                [11]],
            [
                [ 3],
                [ 7],
                [11]]
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_maxTensor3AlongAxes
    func test_maxTensor3AlongAxes() {
        let v = array([
            [
                [10,   2],
                [ 3,   4],
                [ 5,  -6]
            ],
            [
                [ 1,   2],
                [ 3,   4],
                [ 5,   6]
            ]
        ])
        
        // max depths
        let vm = v.max(alongAxes: 0)
        XCTAssert(vm == [
            [
                [10,   2],
                [ 3,   4],
                [ 5,   6]
            ]
        ])

        // max rows
        XCTAssert(v.max(alongAxes: 1) == [
            [
                [10, 4]
            ],
            [
                [ 5, 6]
            ]
        ])

        // max columns
        XCTAssert(v.max(alongAxes: 2) == [
            [
                [10],
                [ 4],
                [ 5]],
            [
                [ 2],
                [ 4],
                [ 6]
            ]
        ])
    }

    //--------------------------------------------------------------------------
    // test_minTensor3AlongAxes
    func test_minTensor3AlongAxes() {
        let v = array([
            [
                [10,   2],
                [ 3,   4],
                [ 5,  -6]
            ],
            [
                [ 1,   2],
                [ 3,   4],
                [ 5,   6]
            ]
        ])
        
        // depths
        XCTAssert(v.min(alongAxes: 0) == [
            [
                [1,  2],
                [3,  4],
                [5, -6]
            ]
        ])
        
        // rows
        XCTAssert(v.min(alongAxes: 1) == [
            [
                [3, -6]
            ],
            [
                [1, 2]
            ]
        ])
        
        // columns
        XCTAssert(v.min(alongAxes: 2) == [
            [
                [ 2],
                [ 3],
                [-6]],
            [
                [1],
                [3],
                [5]
            ]
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_sumTensor1
    func test_sumTensor1() {
        let m = array([0, 1, 2, 3])
        let result = m.sum()
        XCTAssert(result.shape == [1])
        XCTAssert(result.element == 6)
    }

    //--------------------------------------------------------------------------
    // test_sumTensor2
    func test_sumTensor2() {
        let m = array([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        // sum all
        do {
            let result = m.sum()
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15)
        }

        do {
            let result = m.sum(alongAxes: 0, 1)
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15)
        }
        
        // sum cols
        do {
            let result = m.sum(alongAxes: 1)
            XCTAssert(result == [[1], [5], [9]])
        }

        // sum rows
        do {
            let result = m.sum(alongAxes: 0)
            XCTAssert(result == [[6, 9]])
        }
    }

    //--------------------------------------------------------------------------
    // test_abssumTensor2
    func test_abssumTensor2() {
        let m = array([
            [ 0, -1],
            [-2,  3],
            [ 4, -5]
        ])

        // sum all
        do {
            let result = m.abssum()
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15)
        }

        do {
            let result = m.abssum(alongAxes: 0, 1)
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15)
        }
        
        // sum cols
        do {
            let result = m.abssum(alongAxes: 1)
            XCTAssert(result == [[1], [5], [9]])
        }

        // sum rows
        do {
            let result = m.abssum(alongAxes: 0)
            XCTAssert(result == [[6, 9]])
        }
    }

    //--------------------------------------------------------------------------
    // test_allTensor1
    func test_allTensor1() {
        let a = array([true, true, true])
        XCTAssert(a.all().element == true)
        
        let a1 = array([true, false, true])
        XCTAssert(a1.all().element == false)
        
        let a2 = array([false, false, false])
        XCTAssert(a2.all().element == false)
    }
    
    //--------------------------------------------------------------------------
    // test_anyTensor1
    func test_anyTensor1() {
        let a = array([true, true, true])
        XCTAssert(a.any().element == true)
        
        let a1 = array([false, false, true])
        XCTAssert(a1.any().element == true)
        
        let a2 = array([false, false, false])
        XCTAssert(a2.any().element == false)
    }
    
    //--------------------------------------------------------------------------
    // test_maxTensor2
    func test_maxTensor2() {
        let m = array([
            [-1, 3, -6],
            [1, -3,  6],
        ])
        XCTAssert(m.max(alongAxes: 0) == [[1, 3, 6]])
        XCTAssert(m.max(alongAxes: 1) == [[3], [6]])
        XCTAssert(m.max().element == 6)
    }

    //--------------------------------------------------------------------------
    // test_minTensor2
    func test_minTensor2() {
        let m = array([
            [-1,  3, -6],
            [ 1, -3,  6],
        ])
        XCTAssert(m.min(alongAxes: 0) == [[-1, -3, -6]])
        XCTAssert(m.min(alongAxes: 1) == [[-6], [-3]])
        XCTAssert(m.min().element == -6)
    }
    
    //--------------------------------------------------------------------------
    // test_absmaxTensor2
    func test_absmaxTensor2() {
        let m = array([
            [-1,  3, -6],
            [ 1, -3,  6],
        ])
        XCTAssert(m.absmax(alongAxes: 0) == [[1, 3, 6]])
        XCTAssert(m.absmax(alongAxes: 1) == [[6], [6]])
        XCTAssert(m.absmax().element == 6)
    }
        
    //----------------------------------------------------------------------
    // test_meanTensor2
    func test_meanTensor2() {
        let m = array([
            [0, 1],
            [2, 3],
            [4, 5]
        ])
        
        // mean all
        do {
            let result = m.mean()
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15 / 6)
        }
        
        do {
            let result = m.mean(alongAxes: 0, 1)
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == 15 / 6)
        }
        
        // mean cols
        do {
            let result = m.mean(alongAxes: 1)
            XCTAssert(result == [[0.5], [2.5], [4.5]])
        }
        
        // mean rows
        do {
            let result = m.mean(alongAxes: 0)
            XCTAssert(result == [[2, 3]])
        }
    }
    
    //--------------------------------------------------------------------------
    // test_sqrtSumSquaresTensor2
    func test_sqrtSumSquaresTensor2() {
        let m = array([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        // sum all
        do {
            let chained = m.squared().sum().sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == chained.element)
        }

        do {
            let chained = m.squared().sum(alongAxes: 0, 1).sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.shape == [1, 1])
            XCTAssert(result.element == chained.element)
        }
        
        // sum cols
        do {
            let chained = m.squared().sum(alongAxes: 1).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 1)
            XCTAssert(result.shape == [3, 1])
            XCTAssert(result == chained)
        }

        // sum rows
        do {
            let chained = m.squared().sum(alongAxes: 0).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 0)
            XCTAssert(result.shape == [1, 2])
            XCTAssert(result == chained)
        }
    }
    
    //--------------------------------------------------------------------------
    // test_sqrtSumSquaresTensor3
    func test_sqrtSumSquaresTensor3() {
        let m = array([
            [
                [ 0,  1],
                [ 2,  3],
                [ 4,  5]
            ],
            [
                [ 6,  7],
                [ 8,  9],
                [10, 11]
            ]
        ])
        
        // all
        do {
            let chained = m.squared().sum().sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.shape == [1, 1, 1])
            XCTAssert(result == chained)
        }

        do {
            let chained = m.squared().sum(alongAxes: 0, 1, 2).sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.shape == [1, 1, 1])
            XCTAssert(result == chained)
        }
        
        // deps
        do {
            let chained = m.squared().sum(alongAxes: 0).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 0)
            XCTAssert(result.shape == [1, 3, 2])
            XCTAssert(result == chained)
        }

        // rows
        do {
            let chained = m.squared().sum(alongAxes: 1).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 1)
            XCTAssert(result.shape == [2, 1, 2])
            XCTAssert(result == chained)
        }

        // cols
        do {
            let chained = m.squared().sum(alongAxes: 2).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 2)
            XCTAssert(result.shape == [2, 3, 1])
            XCTAssert(result == chained)
        }
    }
}

