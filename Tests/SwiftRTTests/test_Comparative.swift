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
        let a = array([true, false, true, false, true])
        let b = array([false, true, false, true, true])
        XCTAssert((a .&& b) == [false, false, false, false, true])
        XCTAssert((a .|| b) == [true, true, true, true, true])
    }
    
    //--------------------------------------------------------------------------
    // test_elementsAlmostEqual
    func test_elementsAlmostEqual() {
        let a = array([[0, 1.05], [2.0, -3], [4.2, 5.001]])
        let b = array([[0, 1.00], [2.1,  3], [4.0, 4.999]])
        let result = elementsAlmostEqual(a, b, tolerance: 0.1)
        XCTAssert(result == [[true, true], [true, false], [false, true]])
    }
    
    //--------------------------------------------------------------------------
    // test_equality
    func test_equality() {
        // compare by value
        let a = array(0..<6, (3, 2))
        let b = array(0..<6, (3, 2))
        XCTAssert(a == b)

        // compare by value not equal
        let other = array(1..<7, (3, 2))
        XCTAssert(a != other)
        
        // compare via alias detection
        let c = b
        XCTAssert(c == b)
        
        let d = array(1..<7, (3, 2))
        let ne = (d .!= c).any().element
        XCTAssert(ne)
        XCTAssert(d != c)
    }

    //--------------------------------------------------------------------------
    // test_maximum
    func test_max() {
        let a = array([[0, 1], [-2, -3], [-4, 5]])
        let b = array([[0, -7], [2, 3], [4, 5]])
        XCTAssert(max(a, b) == [[0, 1], [2, 3], [4, 5]])
        
        let (ga, gb) = pullback(at: a, b, in: { max($0, $1) })(ones(like: a))
        XCTAssert(ga == [[1, 1], [0, 0], [0, 1]])
        XCTAssert(gb == [[0, 0], [1, 1], [1, 0]])
    }

    //--------------------------------------------------------------------------
    // test_maxScalar
    func test_maxScalar() {
        let a = array(0...5, (3, 2))
        XCTAssert(max(a, 2) == [[2, 2], [2, 3], [4, 5]])
        XCTAssert(max(2, a) == [[2, 2], [2, 3], [4, 5]])
    }

    //--------------------------------------------------------------------------
    // test_min
    func test_min() {
        let a = array([[0,  1], [2, -3], [4, -5]])
        let b = array([[0, -1], [-2, 3], [-4, 5]])
        XCTAssert(min(a, b) == [[0, -1], [-2, -3], [-4, -5]])

        let (ga, gb) = pullback(at: a, b, in: { min($0, $1) })(ones(like: a))
        XCTAssert(ga == [[1, 0], [0, 1], [0, 1]])
        XCTAssert(gb == [[0, 1], [1, 0], [1, 0]])
    }

    //--------------------------------------------------------------------------
    // test_minScalar
    func test_minScalar() {
        let a = array(0...5, (3, 2))
        XCTAssert(min(a, 3) == [[0, 1], [2, 3], [3, 3]])
        XCTAssert(min(3, a) == [[0, 1], [2, 3], [3, 3]])
    }
}
