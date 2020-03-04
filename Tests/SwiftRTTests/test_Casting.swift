//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2
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

class test_Casting: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_stacking", test_stacking),
        ("test_stackingExpression", test_stackingExpression),
        ("test_castElements", test_castElements),
        ("test_flattening", test_flattening),
        ("test_squeezing", test_squeezing),
    ]

    //--------------------------------------------------------------------------
    // test_stacking
    func test_stacking() {
        let m0 = Matrix(2, 3, with: 0..<6)
        let m1 = Matrix(2, 3, with: 6..<12)

        let v0 = Volume(stacking: m0, m1)
        XCTAssert(v0 == 0..<12)

        let v1 = Volume(stacking: m0, m1, alongAxis: 1)
        XCTAssert(v1.array == [
            [[0, 1, 2], [6,  7,  8]],
            [[3, 4, 5], [9, 10, 11]]])

        let v2 = Volume(stacking: m0, m1, alongAxis: 2)
        XCTAssert(v2.array ==
            [[[0, 6],
              [1, 7],
              [2, 8]],
             
             [[3, 9],
              [4, 10],
              [5, 11]]])
    }
    
    //--------------------------------------------------------------------------
    // test_stackingExpression
    func test_stackingExpression() {
        let i = 3
        let j = 3
        let maxK: Float = 16
        
        let k1 = Matrix(5, 6, with: 0..<30)
        let mask = Matrix(squeezing: Volume(stacking: [
            k1[0...j  , 1...i  ],
            k1[0...j  , 2...i+1],
            k1[1...j+1, 1...i  ],
            k1[1...j+1, 2...i+1]
        ]).max(alongAxes: 0)) .<= maxK

        XCTAssert(mask.array == [[true, true, true],
                                 [true, true, true],
                                 [false, false, false],
                                 [false, false, false]])
    }
    
    //--------------------------------------------------------------------------
    // test_castElements
    func test_castElements() {
        let fMatrix = Matrix(3, 2, with: 0..<6)
        let iMatrix = IndexMatrix(fMatrix)
        XCTAssert(iMatrix == 0..<6)
    }

    //--------------------------------------------------------------------------
    // test_flattening
    func test_flattening() {
        let volume = Volume(2, 3, 4, with: 0..<24)
        
        // volume to matrix
        let matrix = Matrix(flattening: volume)
        XCTAssert(matrix.bounds == [2, 12])

        // noop matrix to matrix
        let m2 = Matrix(flattening: matrix)
        XCTAssert(m2.bounds == [2, 12])

        // volume to vector
        let v1 = Vector(flattening: volume)
        XCTAssert(v1.bounds == [24])

        // matrix to vector
        let v2 = Vector(flattening: matrix)
        XCTAssert(v2.bounds == [24])
        
        do {
            let ones = Matrix(repeating: 1, to: (2, 12))
            let g = pullback(at: volume, in: { Matrix(flattening: $0) })(ones)
            XCTAssert(g == [Float](repeating: 1, count: 24))
        }
    }
    
    //--------------------------------------------------------------------------
    // test_squeezing
    func test_squeezing() {
        let volume = Volume(2, 3, 4, with: 0..<24)

        let sumVolumeCols = volume.sum(alongAxes: 2)
        XCTAssert(sumVolumeCols.bounds == [2, 3, 1])
        let m0 = Matrix(squeezing: sumVolumeCols)
        XCTAssert(m0.bounds == [2, 3])
        
        let sumVolumeRows = volume.sum(alongAxes: 1)
        XCTAssert(sumVolumeRows.bounds == [2, 1, 4])
        let m2 = Matrix(squeezing: sumVolumeRows, alongAxes: 1)
        XCTAssert(m2.bounds == [2, 4])
        
        // test negative axes
        let m3 = Matrix(squeezing: sumVolumeRows, alongAxes: -2)
        XCTAssert(m3.bounds == [2, 4])
        
        do {
            let ones = Matrix(repeating: 1, to: (2, 12))
            let g = pullback(at: sumVolumeRows,
                             in: { Matrix(squeezing: $0, alongAxes: 1) })(ones)
            XCTAssert(g == [Float](repeating: 1, count: 24))
        }
    }
}
