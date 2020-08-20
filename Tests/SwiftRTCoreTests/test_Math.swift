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
        ("test_exp", test_exp),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_sign", test_sign),
        ("test_squared", test_squared),
    ]
    
    //--------------------------------------------------------------------------
    // test_abs
    func test_abs() {
//        Context.log.level = .diagnostic
        // integer abs
        let a = array([-1, 2, -3, 4], type: Int32.self)
        XCTAssert(abs(a) == [1, 2, 3, 4])

        // real abs
        let b = array([-1.0, 2, -3, 4])
        let g = pullback(at: b, in: { abs($0) })(ones(like: b))
        XCTAssert(g == [-1, 1, -1, 1])
    }
    
    //--------------------------------------------------------------------------
    // test_exp
    func test_exp() {
        let a = array([0.0, 1, 2, 3, 4, 5])
        let expected = a.map(Foundation.exp)
        XCTAssert(exp(a) == expected)
        
        let b = array([1.0, 2, 3])
        let g = pullback(at: b, in: { exp($0) })(ones(like: b))
        let e = array([2.7182817,  7.389056, 20.085537])
        XCTAssert(elementsAlmostEqual(g, e, tolerance: 0.0001).all().element)
    }

    //--------------------------------------------------------------------------
    // test_log
    func test_log() {
        let a = array([0.0, 1, 2, 3, 4, 5], (3, 2))
        let expected = a.map(Foundation.log)
        XCTAssert(log(a).flatArray == expected)

        let b = array([1.0, -2.0, 3.0])
        let g = pullback(at: b, in: { log($0) })(ones(like: b))
        let e = array([1.0, -0.5, 0.33333334])
        XCTAssert(elementsAlmostEqual(g, e, tolerance: 0.0001).all().element)
    }

    //--------------------------------------------------------------------------
    // test_neg
    func test_neg() {
        let a = array(0..<6, (3, 2))
        let expected = a.map(-)
        XCTAssert(neg(a).flatArray == expected)

        let b = -a
        XCTAssert(b.flatArray == expected)

        let c = array([1.0, -2.0, 3.0])
        let g = pullback(at: c, in: { (-$0) })(ones(like: c))
        XCTAssert(g == [-1, -1, -1])
    }

    //--------------------------------------------------------------------------
    // test_sign
    func test_sign() {
        let a = array([-1, 2, -3, 4])
        XCTAssert(sign(a) == [-1, 1, -1, 1])

        let b = array([-1.0, 2.0, -3.0, 4.0])
        let g = pullback(at: b, in: { sign($0) })(ones(like: b))
        XCTAssert(g == [0, 0, 0, 0])
    }

    //--------------------------------------------------------------------------
    // test_squared
    func test_squared() {
        let a = array([[0, -1], [2, -3], [4, 5]])
        XCTAssert(a.squared() == [[0, 1], [4, 9], [16, 25]])

        let b = array([1.0, -2.0, 3.0])
        let g = pullback(at: b, in: { $0.squared() })(ones(like: b))
        XCTAssert(g == [2, -4, 6])
    }
}
