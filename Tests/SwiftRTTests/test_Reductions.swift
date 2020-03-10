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
        ("test_sumVolumeAlongAxes", test_sumVolumeAlongAxes),
        ("test_minVolumeAlongAxes", test_minVolumeAlongAxes),
        ("test_maxVolumeAlongAxes", test_maxVolumeAlongAxes),
        ("test_sumVector", test_sumVector),
        ("test_sumMatrix", test_sumMatrix),
        ("test_abssumMatrix", test_abssumMatrix),
        ("test_allVector", test_allVector),
        ("test_anyVector", test_anyVector),
        ("test_meanMatrix", test_meanMatrix),
        ("test_maxMatrix", test_maxMatrix),
        ("test_minMatrix", test_minMatrix),
        ("test_absmaxMatrix", test_absmaxMatrix),
        ("test_sqrtSumSquaresMatrix", test_sqrtSumSquaresMatrix),
        ("test_sqrtSumSquaresVolume", test_sqrtSumSquaresVolume),
    ]

    //--------------------------------------------------------------------------
    // test_sumVolumeAlongAxes
    func test_sumVolumeAlongAxes() {
        let v = IndexVolume([
            [
                [10,   2],
                [ 3,   4],
                [ 5,   6]
            ],
            [
                [ 1,   2],
                [ 3,   4],
                [ 5,   6]
            ]
        ])

        // sum depths
        XCTAssert(v.sum(alongAxes: 0).array == [
            [
                [11,  4],
                [ 6,  8],
                [10, 12]
            ]
        ])
        
        // sum rows
        XCTAssert(v.sum(alongAxes: 1).array == [
            [
                [18, 12]
            ],
            [
                [ 9, 12]
            ]
        ])

        // sum columns
        XCTAssert(v.sum(alongAxes: 2).array == [
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
    // test_maxVolumeAlongAxes
    func test_maxVolumeAlongAxes() {
        let v = IndexVolume([
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
        XCTAssert(vm.array == [
            [
                [10,   2],
                [ 3,   4],
                [ 5,   6]
            ]
        ])

        // max rows
        XCTAssert(v.max(alongAxes: 1).array == [
            [
                [10, 4]
            ],
            [
                [ 5, 6]
            ]
        ])

        // max columns
        XCTAssert(v.max(alongAxes: 2).array == [
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
    // test_minVolumeAlongAxes
    func test_minVolumeAlongAxes() {
        let v = IndexVolume([
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
        XCTAssert(v.min(alongAxes: 0).array == [
            [
                [1,  2],
                [3,  4],
                [5, -6]
            ]
        ])
        
        // rows
        XCTAssert(v.min(alongAxes: 1).array == [
            [
                [3, -6]
            ],
            [
                [1, 2]
            ]
        ])
        
        // columns
        XCTAssert(v.min(alongAxes: 2).array == [
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
    // test_sumVector
    func test_sumVector() {
        let m = Vector([0, 1, 2, 3])
        let result = m.sum()
        XCTAssert(result.bounds == [1])
        XCTAssert(result.element == 6)
    }

    //--------------------------------------------------------------------------
    // test_sumMatrix
    func test_sumMatrix() {
        let m = Matrix(3, 2, with: [
            0, 1,
            2, 3,
            4, 5
        ])

        // sum all
        do {
            let result = m.sum()
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15)
        }

        do {
            let result = m.sum(alongAxes: 0, 1)
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15)
        }
        
        // sum cols
        do {
            let result = m.sum(alongAxes: 1)
            XCTAssert(result.bounds == [3, 1])
            XCTAssert(result == [
                1,
                5,
                9
            ])
        }

        // sum rows
        do {
            let result = m.sum(alongAxes: 0)
            XCTAssert(result.bounds == [1, 2])
            XCTAssert(result == [
                6, 9
            ])
        }
    }

    //--------------------------------------------------------------------------
    // test_abssumMatrix
    func test_abssumMatrix() {
        let m = Matrix(3, 2, with: [
             0, -1,
            -2,  3,
             4, -5
        ])

        // sum all
        do {
            let result = m.abssum()
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15)
        }

        do {
            let result = m.abssum(alongAxes: 0, 1)
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15)
        }
        
        // sum cols
        do {
            let result = m.abssum(alongAxes: 1)
            XCTAssert(result.bounds == [3, 1])
            XCTAssert(result == [
                1,
                5,
                9
            ])
        }

        // sum rows
        do {
            let result = m.abssum(alongAxes: 0)
            XCTAssert(result.bounds == [1, 2])
            XCTAssert(result == [
                6, 9
            ])
        }
    }

    //--------------------------------------------------------------------------
    // test_allVector
    func test_allVector() {
        let m0 = BoolVector([true, true, true])
        XCTAssert(m0.all().element == true)
        
        let m1 = BoolVector([true, false, true])
        XCTAssert(m1.all().element == false)
        
        let m2 = BoolVector([false, false, false])
        XCTAssert(m2.all().element == false)
    }
    
    //--------------------------------------------------------------------------
    // test_anyVector
    func test_anyVector() {
        let m0 = BoolVector([true, true, true])
        XCTAssert(m0.any().element == true)
        
        let m1 = BoolVector([false, false, true])
        XCTAssert(m1.any().element == true)
        
        let m2 = BoolVector([false, false, false])
        XCTAssert(m2.any().element == false)
    }
    
    //--------------------------------------------------------------------------
    // test_maxMatrix
    func test_maxMatrix() {
        let m = Matrix([
            [-1, 3, -6],
            [1, -3,  6],
        ])
        XCTAssert(m.max(alongAxes: 0) == [1, 3, 6])
        XCTAssert(m.max(alongAxes: 1) == [3, 6])
        XCTAssert(m.max().element == 6)
    }

    //--------------------------------------------------------------------------
    // test_minMatrix
    func test_minMatrix() {
        let m = Matrix([
            [-1,  3, -6],
            [ 1, -3,  6],
        ])
        XCTAssert(m.min(alongAxes: 0) == [-1, -3, -6])
        XCTAssert(m.min(alongAxes: 1) == [-6, -3])
        XCTAssert(m.min().element == -6)
    }
    
    //--------------------------------------------------------------------------
    // test_absmaxMatrix
    func test_absmaxMatrix() {
        let m = Matrix([
            [-1,  3, -6],
            [ 1, -3,  6],
        ])
        XCTAssert(m.absmax(alongAxes: 0) == [1, 3, 6])
        XCTAssert(m.absmax(alongAxes: 1) == [6, 6])
        XCTAssert(m.absmax().element == 6)
    }
        
    //----------------------------------------------------------------------
    // test_meanMatrix
    func test_meanMatrix() {
        let m = Matrix(3, 2, with: [
            0, 1,
            2, 3,
            4, 5
        ])
        
        // mean all
        do {
            let result = m.mean()
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15 / 6)
        }
        
        do {
            let result = m.mean(alongAxes: 0, 1)
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == 15 / 6)
        }
        
        // mean cols
        do {
            let result = m.mean(alongAxes: 1)
            XCTAssert(result.bounds == [3, 1])
            XCTAssert(result == [0.5, 2.5, 4.5])
        }
        
        // mean rows
        do {
            let result = m.mean(alongAxes: 0)
            XCTAssert(result.bounds == [1, 2])
            XCTAssert(result == [2, 3])
        }
    }
    
    //--------------------------------------------------------------------------
    // test_sqrtSumSquaresMatrix
    func test_sqrtSumSquaresMatrix() {
        let m = Matrix([
            [0, 1],
            [2, 3],
            [4, 5]
        ])

        // sum all
        do {
            let chained = m.squared().sum().sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == chained.element)
        }

        do {
            let chained = m.squared().sum(alongAxes: 0, 1).sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.bounds == [1, 1])
            XCTAssert(result.element == chained.element)
        }
        
        // sum cols
        do {
            let chained = m.squared().sum(alongAxes: 1).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 1)
            XCTAssert(result.bounds == [3, 1])
            XCTAssert(result == chained)
        }

        // sum rows
        do {
            let chained = m.squared().sum(alongAxes: 0).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 0)
            XCTAssert(result.bounds == [1, 2])
            XCTAssert(result == chained)
        }
    }
    
    //--------------------------------------------------------------------------
    // test_sqrtSumSquaresVolume
    func test_sqrtSumSquaresVolume() {
        let m = Volume([
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
            XCTAssert(result.bounds == [1, 1, 1])
            XCTAssert(result == chained)
        }

        do {
            let chained = m.squared().sum(alongAxes: 0, 1, 2).sqrt()
            let result = m.sqrtSumSquares()
            XCTAssert(result.bounds == [1, 1, 1])
            XCTAssert(result == chained)
        }
        
        // deps
        do {
            let chained = m.squared().sum(alongAxes: 0).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 0)
            XCTAssert(result.bounds == [1, 3, 2])
            XCTAssert(result == chained)
        }

        // rows
        do {
            let chained = m.squared().sum(alongAxes: 1).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 1)
            XCTAssert(result.bounds == [2, 1, 2])
            XCTAssert(result == chained)
        }

        // cols
        do {
            let chained = m.squared().sum(alongAxes: 2).sqrt()
            let result = m.sqrtSumSquares(alongAxes: 2)
            XCTAssert(result.bounds == [2, 3, 1])
            XCTAssert(result == chained)
        }
    }
}

