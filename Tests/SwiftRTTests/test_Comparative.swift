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

class test_Comparative: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_elementWiseAndOr", test_elementWiseAndOr),
        ("test_elementsAlmostEqual", test_elementsAlmostEqual),
        ("test_equality", test_equality),
        ("test_max", test_max),
        ("test_maxScalar", test_maxScalar),
        ("test_min", test_min),
        ("test_minScalar", test_minScalar),
    ]

    //--------------------------------------------------------------------------
    // test_elementsAlmostEqual
    func test_elementWiseAndOr() {
        let v0 = BoolVector(with: [true, false, true, false, true])
        let v1 = BoolVector(with: [false, true, false, true, true])
        XCTAssert((v0 .&& v1) == [false, false, false, false, true])
        XCTAssert((v0 .|| v1) == [true, true, true, true, true])
    }
    
    //--------------------------------------------------------------------------
    // test_elementsAlmostEqual
    func test_elementsAlmostEqual() {
        let m1 = Matrix(3, 2, with: [0, 1.05, 2.0, -3, 4.2, 5.001])
        let m2 = Matrix(3, 2, with: [0, 1.00, 2.1,  3, 4.0, 4.999])
        let expected = [true, true, true, false, false, true]
        let result = elementsAlmostEqual(m1, m2, tolerance: 0.1)
        XCTAssert(result.flatArray == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_equality
    func test_equality() {
        // compare by value
        let m1 = Matrix(3, 2, with: 0..<6)
        let m2 = Matrix(3, 2, with: 0..<6)
        XCTAssert(m1 == m2)

        // compare by value not equal
        let other = Matrix(3, 2, with: 1..<7)
        XCTAssert(m1 != other)
        
        // compare via alias detection
        let m3 = m2
        XCTAssert(m3 == m2)
        
        let m4 = Matrix(3, 2, with: 1..<7)
        let ne = (m4 .!= m3).any().element
        XCTAssert(ne)
        XCTAssert(m4 != m3)
    }

    //--------------------------------------------------------------------------
    // test_maximum
    func test_max() {
        let m1 = Matrix(3, 2, with: [0, 1, -2, -3, -4, 5])
        let m2 = Matrix(3, 2, with: [0, -7, 2, 3, 4, 5])
        XCTAssert(max(m1, m2) == [0, 1, 2, 3, 4, 5])
        
//        let ones = Matrix(repeating: 1, like: m1)
//        let (gm1, gm2) = pullback(at: m1, m2, in: { max($0, $1) })(ones)
//        XCTAssert(gm1 == [1, 1, 0, 0, 0, 1])
//        XCTAssert(gm2 == [0, 0, 1, 1, 1, 0])
    }
    
    //--------------------------------------------------------------------------
    // test_maxScalar
    func test_maxScalar() {
        let m1 = Matrix(3, 2, with: 0...5)
        let scalar: Float = 2
        let expected = [2, 2, 2, 3, 4, 5]
        XCTAssert(max(m1, scalar) == expected)
        XCTAssert(max(scalar, m1) == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_min
    func test_min() {
        let m1 = Matrix(3, 2, with: [0, 1, 2, -3, 4, -5])
        let m2 = Matrix(3, 2, with: [0, -1, -2, 3, -4, 5])
        XCTAssert(min(m1, m2) == [0, -1, -2, -3, -4, -5])

//        let ones = Matrix(repeating: 1, like: m1)
//        let (gm1, gm2) = pullback(at: m1, m2, in: { min($0, $1) })(ones)
//        XCTAssert(gm1 == [1, 0, 0, 1, 0, 1])
//        XCTAssert(gm2 == [0, 1, 1, 0, 1, 0])
    }
    
    //--------------------------------------------------------------------------
    // test_minScalar
    func test_minScalar() {
        let m1 = Matrix(3, 2, with: 0...5)
        let scalar: Float = 3
        let expected = [0, 1, 2, 3, 3, 3]
        XCTAssert(min(m1, scalar) == expected)
        XCTAssert(min(scalar, m1) == expected)
    }
}
