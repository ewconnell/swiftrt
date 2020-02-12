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

class test_Math: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_abs", test_abs),
        ("test_concat", test_concat),
        ("test_exp", test_exp),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_sign", test_sign),
        ("test_squared", test_squared),
    ]
    
    //--------------------------------------------------------------------------
    // test_abs
    func test_abs() {
        let v = Vector(with: [-1, 2, -3, 4])
        XCTAssert(abs(v) == [1, 2, 3, 4])
        
        let g = gradient(at: v, in: { abs($0).sum() })
        XCTAssert(g == [-1, 1, -1, 1])
    }

    //--------------------------------------------------------------------------
    // test_concat
    func test_concat() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c1 = t1.concat(t2)
        XCTAssert(c1.extents == [4, 3])
        XCTAssert(c1 == [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ])
        
        let c2 = t1.concat(t2, alongAxis: 1)
        XCTAssert(c2.extents == [2, 6])
        XCTAssert(c2 == [
            1, 2, 3,  7,  8,  9,
            4, 5, 6, 10, 11, 12
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_exp
    func test_exp() {
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let values = exp(matrix)
        let expected: [Float] = range.map { Foundation.exp(Float($0)) }
        XCTAssert(values == expected)
        
        let v = Vector(with: 1...3)
        let g = gradient(at: v, in: { exp($0).sum() })
        let e = Vector(with: [2.7182817,  7.389056, 20.085537])
        XCTAssert(elementsAlmostEqual(g, e, tolerance: 0.0001).all().element)
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let values = log(matrix)
        let expected: [Float] = range.map { Foundation.log(Float($0)) }
        XCTAssert(values == expected)
        
        let v = Vector(with: [1, -2, 3])
        let g = gradient(at: v, in: { log($0).sum() })
        let e = Vector(with: [1.0, -0.5, 0.33333334])
        XCTAssert(elementsAlmostEqual(g, e, tolerance: 0.0001).all().element)
    }
    
    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let range = 0..<6
        let matrix = Matrix(3, 2, with: range)
        let expected: [Float] = range.map { -Float($0) }

        let values = matrix.neg()
        XCTAssert(values == expected)
        
        let values2 = -matrix
        XCTAssert(values2 == expected)
        
        let v = Vector(with: [1, -2, 3])
        let g = gradient(at: v, in: { (-$0).sum() })
        XCTAssert(g == [-1, -1, -1])
    }
    
    //--------------------------------------------------------------------------
    // test_sign
    func test_sign() {
        let v = Vector(with: [-1, 2, -3, 4])
        XCTAssert(sign(v) == [-1, 1, -1, 1])
        
        let g = gradient(at: v, in: { sign($0).sum() })
        XCTAssert(g == [0, 0, 0, 0])
    }

    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let matrix = Matrix(3, 2, with: [0, -1, 2, -3, 4, 5])
        let values = matrix.squared()
        let expected: [Float] = (0...5).map { Float($0 * $0) }
        XCTAssert(values == expected)
        
        let v = Vector(with: [1, -2, 3])
        let g = gradient(at: v, in: { $0.squared().sum() })
        XCTAssert(g == [2, -4, 6])
    }
}
