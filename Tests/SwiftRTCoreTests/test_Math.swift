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
        ("test_multiplyAdd", test_multiplyAdd),
        ("test_juliaMath", test_juliaMath),
        ("test_abs", test_abs),
        ("test_atan2", test_atan2),
        ("test_erf", test_erf),
        ("test_exp", test_exp),
        ("test_log", test_log),
        ("test_neg", test_neg),
        ("test_sign", test_sign),
        ("test_squared", test_squared),
    ]

    //--------------------------------------------------------------------------
    func test_multiplyAdd() {
        do {
            let a = array(0..<3)
            let b = array([2, 2, 2])
            let c = multiply(a, a, add: b)
            XCTAssert(c == [2, 3, 6])
        }

        do {
            let a = array(0..<3)
            let b: Float = 2
            let c = multiply(a, a, add: b)
            XCTAssert(c == [2, 3, 6])
        }
    }

    //--------------------------------------------------------------------------
    func test_juliaMath() {
        typealias CF = Complex<Float>
        let iterations = 3
        let size = (r: 5, c: 5)
        let tolerance: Float = 4.0
        let C = CF(-0.8, 0.156)
        let first = CF(-1.7, 1.7)
        let last = CF(1.7, -1.7)
        let rFirst = CF(first.real, 0), rLast = CF(last.real, 0)
        let iFirst = CF(0, first.imaginary), iLast = CF(0, last.imaginary)

        // repeat rows of real range, columns of imaginary range, and combine
        var Z = repeating(array(from: rFirst, to: rLast, (1, size.c)), size) +
                repeating(array(from: iFirst, to: iLast, (size.r, 1)), size)
        
        XCTAssert(Z == [
            [CF(-1.7, 1.7), CF(-0.85, 1.7), CF(0.0, 1.7), CF(0.85000014, 1.7), CF(1.7, 1.7)],
            [CF(-1.7, 0.85), CF(-0.85, 0.85), CF(0.0, 0.85), CF(0.85000014, 0.85), CF(1.7, 0.85)],
            [CF(-1.7, 0.0), CF(-0.85, 0.0), CF(0.0, 0.0), CF(0.85000014, 0.0), CF(1.7, 0.0)],
            [CF(-1.7, -0.85000014), CF(-0.85, -0.85000014), CF(0.0, -0.85000014), CF(0.85000014, -0.85000014), CF(1.7, -0.85000014)],
            [CF(-1.7, -1.7), CF(-0.85, -1.7), CF(0.0, -1.7), CF(0.85000014, -1.7), CF(1.7, -1.7)]
        ])
        
        
        var divergence = full(size, iterations)
        XCTAssert(divergence == [
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0]
        ])
        
        // Z = Z * Z + C
        Z = multiply(Z, Z, add: C)
        let expected = array([
            [CF(-0.8, -5.624), CF(-2.9675, -2.7340002), CF(-3.69, 0.156), CF(-2.9674997, 3.0460005), CF(-0.8, 5.9360003)],
            [CF(1.3675001, -2.7340002), CF(-0.8, -1.289), CF(-1.5225, 0.156), CF(-0.7999998, 1.6010003), CF(1.3675001, 3.046)],
            [CF(2.0900002, 0.156), CF(-0.077499986, 0.156), CF(-0.8, 0.156), CF(-0.07749975, 0.156), CF(2.0900002, 0.156)],
            [CF(1.3674998, 3.0460005), CF(-0.80000025, 1.6010003), CF(-1.5225003, 0.156), CF(-0.8, -1.2890005), CF(1.3674998, -2.7340007)],
            [CF(-0.8, 5.9360003), CF(-2.9675, 3.046), CF(-3.69, 0.156), CF(-2.9674997, -2.7340007), CF(-0.8, -5.624)]
        ])
        XCTAssert(elementsAlmostEqual(Z, expected, tolerance: CF(0.00001, 0.00001)).all().element)
        
        divergence[abs(Z) .> tolerance] = min(divergence, 0)
        XCTAssert(divergence == [
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 3.0, 0.0, 0.0]
        ])
    }
    
    //--------------------------------------------------------------------------
    func test_abs() {
        // Int32 abs
        let a = array([-1, 2, -3, 4, -5], type: Int32.self)
        XCTAssert(abs(a) == [1, 2, 3, 4, 5])

        // Float16 abs
        let f16 = array([-1, 2, -3, 4, -5], type: Float16.self)
        XCTAssert(abs(f16) == [1, 2, 3, 4, 5])

        // Float abs
        let b = array([-1, 2, -3, 4, -5])
        XCTAssert(abs(b) == [1, 2, 3, 4, 5])
        
        let g = pullback(at: b, in: { abs($0) })(ones(like: b))
        XCTAssert(g == [-1, 1, -1, 1, -1])

        // Complex
        let c = abs(Complex<Float>(3, 4))
        XCTAssert(c == 5)
    }

    //--------------------------------------------------------------------------
    // test_atan2
    func test_atan2() {
        let a = array([[1, 2], [3, 4], [5, 6]])
        let b = array([[1, -2], [-3, 4], [5, -6]])
        let result = atan2(y: a, x: b)
        XCTAssert(result == [[0.7853982, 2.3561945], [2.3561945, 0.7853982], [0.7853982, 2.3561945]])

        let (da, db) = pullback(at: a, b, in: { atan2(y: $0, x: $1) })(ones(like: result))
        XCTAssert(da == [[0.5, -0.25], [-0.16666667, 0.125], [0.099999994, -0.083333336]])
        XCTAssert(db == [[-0.5, -0.25], [-0.16666667, -0.125], [-0.099999994, -0.083333336]])
    }

    //--------------------------------------------------------------------------
    // test_erf
    func test_erf() {
        let a = array([[0, -1], [2, -3], [4, 5]])
        XCTAssert(erf(a) == [[0.0, -0.8427008], [0.9953223, -0.9999779], [1.0, 1.0]])

        let g = pullback(at: a, in: { erf($0) })(ones(like: a))
        let e = array([[1.1283792, 0.41510752], [0.020666987, 0.00013925305],
                       [1.2698236e-07, 1.5670867e-11]])
        XCTAssert(elementsAlmostEqual(g, e, tolerance: 0.0001).all().element)
    }
    
    //--------------------------------------------------------------------------
    // test_exp
    func test_exp() {
        let a = array(0..<6)
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
