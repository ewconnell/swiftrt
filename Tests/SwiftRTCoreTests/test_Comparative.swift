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
import Numerics

class test_Comparative: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        // ("test_compareFloat16", test_compareFloat16),
        // ("test_compareInt8", test_compareInt8),
        // ("test_replace", test_replace),
        // ("test_complexOrder", test_complexOrder),
        // ("test_elementWiseAndOr", test_elementWiseAndOr),
        // ("test_elementsAlmostEqual", test_elementsAlmostEqual),

// these require reductions to work
        // ("test_boolEquality", test_boolEquality),
        // ("test_equality", test_equality),
        
// requires vjpMax to be implemented on gpu
        ("test_max", test_max),
        // ("test_maxScalar", test_maxScalar),
        // ("test_min", test_min),
        // ("test_minScalar", test_minScalar),
    ]

    override func setUpWithError() throws {
        log.level = .diagnostic
    }

    override func tearDownWithError() throws {
        // log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_compareFloat16() {
        let a = array([0, 1, 2], type: Float16.self)
        let b = array([1, 0, 2], type: Float16.self)
        let x = a .> b
        XCTAssert(x == [false, true, false])
    }

    //--------------------------------------------------------------------------
    func test_compareInt8() {
        let a = array([0, 1, 2], type: Int8.self)
        let b = array([1, 0, 2], type: Int8.self)
        let x = a .> b
        XCTAssert(x == [false, true, false])
    }

    //--------------------------------------------------------------------------
    func test_replace() {
        do {
            let a = array([1, 1, 2, 2], type: Int8.self)
            let b = array([0, 0, 2, 1], type: Int8.self)
            let c = array([true, false, false, true])
            let x = replace(x: a, with: b, where: c)
            XCTAssert(x == [0, 1, 2, 1])
        }

        do {
            let a = array(1..<7, (2, 3), type: Int8.self)
            let b = zeros((2, 3), type: Int8.self)
            let c = array([[false, true, false], [true, false, true]])
            let x = replace(x: a, with: b, where: c)
            XCTAssert(x == [[1, 0, 3],[0, 5, 0]])
        }
    }

    //--------------------------------------------------------------------------
    func test_complexOrder() {
        typealias CF = Complex<Float>
        do {
            let a = array(from: CF(0), to: CF(2), count: 5)
            let b = array(from: CF(0.5), to: CF(1), count: 5)
            let x = a .> b
            XCTAssert(x == [false, false, true, true, true])
        }

        do {
            let a = array(from: CF(0), to: CF(2), count: 5)
            let b = array(from: CF(0.5), to: CF(1), count: 5)
            let x = a .< b
            XCTAssert(x == [true, true, false, false, false])
        }

        do {
            let a = array([CF(1), CF(1.5), CF(2)])
            let b = array([CF(0), CF(1.5), CF(2.5)])
            let x = a .>= b
            XCTAssert(x == [true, true, false])
        }

        do {
            let a = array([CF(1), CF(1.5), CF(2)])
            let b = array([CF(0), CF(1.5), CF(2.5)])
            let x = a .<= b
            XCTAssert(x == [false, true, true])
        }
    }
    
    //--------------------------------------------------------------------------
    func test_elementWiseAndOr() {
        let a = array([true, false, true, false, true])
        let b = array([false, true, false, true, true])
        XCTAssert((a .&& b) == [false, false, false, false, true])
        XCTAssert((a .|| b) == [true, true, true, true, true])
    }
    
    //--------------------------------------------------------------------------
    func test_elementsAlmostEqual() {
        let a = array([[0, 1.05], [2.0, -3], [4.2, 5.001]])
        let b = array([[0, 1.00], [2.1,  3], [4.0, 4.999]])
        let result = elementsAlmostEqual(a, b, tolerance: 0.1)
        XCTAssert(result == [[true, true], [true, false], [false, true]])
    }
    
    //--------------------------------------------------------------------------
    func test_boolEquality() {
        let a = array([1, 2, 3, 4, 5], name: "A")
        let b = array([1, 2, 3, 4, 5], name: "B")
        XCTAssert(a == b)
    }
    
    //--------------------------------------------------------------------------
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
    func test_max() {
        let a = array([[0, 1], [-2, -3], [-4, 5]])
        let b = array([[0, -7], [2, 3], [4, 5]])
        XCTAssert(max(a, b) == [[0, 1], [2, 3], [4, 5]])
        XCTAssert(max(a, -2) == [[0, 1], [-2, -2], [-2, 5]])
        
        // both
        let one = ones(like: a)
        let (ga, gb) = pullback(at: a, b, in: { max($0, $1) })(one)
        XCTAssert(ga == [[1, 1], [0, 0], [0, 1]])
        XCTAssert(gb == [[0, 0], [1, 1], [1, 0]])

        // lhs
        let gl = pullback(at: a, in: { max($0, -2) })(one)
        XCTAssert(gl == [[1, 1], [1, 0], [0, 1]])

        // rhs
        let gr = pullback(at: a, in: { max(-2, $0) })(one)
        XCTAssert(gr == [[1, 1], [1, 0], [0, 1]])
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

        // both
        let one = ones(like: a)
        let (ga, gb) = pullback(at: a, b, in: { min($0, $1) })(one)
        XCTAssert(ga == [[1, 0], [0, 1], [0, 1]])
        XCTAssert(gb == [[0, 1], [1, 0], [1, 0]])

        // lhs
        let gl = pullback(at: a, in: { min($0, -2) })(one)
        XCTAssert(gl == [[0, 0], [0, 1], [0, 1]])
        
        // rhs
        let gr = pullback(at: a, in: { min(-2, $0) })(one)
        XCTAssert(gr == [[0, 0], [0, 1], [0, 1]])
    }

    //--------------------------------------------------------------------------
    // test_minScalar
    func test_minScalar() {
        let a = array(0...5, (3, 2))
        XCTAssert(min(a, 3) == [[0, 1], [2, 3], [3, 3]])
        XCTAssert(min(3, a) == [[0, 1], [2, 3], [3, 3]])
    }
}
