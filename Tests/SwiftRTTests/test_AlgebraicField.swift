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

class test_AlgebraicField: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_addSubMulDivComplex", test_addSubMulDivComplex),
        ("test_add", test_add),
        ("test_addInt32", test_addInt32),
        ("test_addUInt8", test_addUInt8),
        ("test_addScalar", test_addScalar),
        ("test_addAndAssign", test_addAndAssign),

        ("test_subtract", test_subtract),
        ("test_subtractScalar", test_subtractScalar),
        ("test_subtractVector", test_subtractVector),
        ("test_subtractAndAssign", test_subtractAndAssign),

        ("test_mul", test_mul),
//        ("test_mulScalar", test_mulScalar),
//        ("test_mulAndAssign", test_mulAndAssign),
//
//        ("test_div", test_div),
//        ("test_divScalar", test_divScalar),
//        ("test_divAndAssign", test_divAndAssign),
    ]
    
    //--------------------------------------------------------------------------
    // test_add
    func test_add() {
        let a = array([[0, 1], [2, 3], [4, 5]])
        let b = array(0..<6, (3, 2))
        let result = a + b
        XCTAssert(result == [[0, 2], [4, 6], [8, 10]])

        let (g1, g2) = pullback(at: a, b, in: { $0 + $1 })(ones(like: a))
        XCTAssert(g1.flat == [1, 1, 1, 1, 1, 1])
        XCTAssert(g2.flat == [1, 1, 1, 1, 1, 1])
    }

    //--------------------------------------------------------------------------
    // test_addInt32
    func test_addInt32() {
        let a = array(0..<6, (3, 2), dtype: Int32.self)
        let b = array(0..<6, (3, 2), dtype: Int32.self)
        let result = a + b
        XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
    }

    //--------------------------------------------------------------------------
    // test_addUInt8
    func test_addUInt8() {
        let a = array(0..<6, (3, 2), dtype: UInt8.self)
        let b = array(0..<6, (3, 2), dtype: UInt8.self)
        let result = a + b
        XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
    }

    //--------------------------------------------------------------------------
    // test_addScalar
    func test_addScalar() {
        let a = array(1...6, (3, 2))
        let result = a + 1
        let expected: [[Float]] = [[2, 3], [4, 5], [6, 7]]
        XCTAssert(result == expected)

        let result2 = 1 + a
        XCTAssert(result2 == expected)
    }

    //--------------------------------------------------------------------------
    // test_addAndAssign
    func test_addAndAssign() {
        var a = array(0...5, (3, 2))
        a += 2
        XCTAssert(a == [[2, 3], [4, 5], [6, 7]])
    }

    //--------------------------------------------------------------------------
    // test_addSubMulDivComplex
    func test_addSubMulDivComplex() {
        typealias CF = Complex<Float>
        let data: [Complex<Float>] = [1, 2, 3, 4]
        let a = array(data, (2, 2))
        let b = array(data, (2, 2))
        let v = ones(like: a)

        // add a scalar
        XCTAssert((a + 1) == [[2, 3], [4, 5]])

        // add tensors
        XCTAssert((a + b) == [[2, 4], [6, 8]])

        // subtract a scalar
        XCTAssert((a - 1) == [[0, 1], [2, 3]])

        // subtract tensors
        XCTAssert((a - b) == [[0, 0], [0, 0]])

        // mul a scalar
        XCTAssert((a * 2) == [[2, 4], [6, 8]])

        // mul tensors
        XCTAssert((a * b) == [[1, 4], [9, 16]])

        // divide by a scalar
        let divExpected = [[CF(0.5), CF(1)], [CF(1.5), CF(2)]]
        XCTAssert((a / 2) == divExpected)

        // divide by a tensor
        XCTAssert((a / b) == [[1, 1], [1, 1]])

        // test add derivative
        do {
            let (g1, g2) = pullback(at: a, b, in: { $0 + $1 })(v)
            XCTAssert(g1 == [[1, 1], [1, 1]])
            XCTAssert(g2 == [[1, 1], [1, 1]])
        }

        do {
            let (g1, g2) = pullback(at: a, b, in: { $0 - $1 })(v)
            XCTAssert(g1 == [[1, 1], [1, 1]])
            XCTAssert(g2 == [[-1, -1], [-1, -1]])
        }
        do {
            let (g1, g2) = pullback(at: a, b, in: { $0 * $1 })(v)
            XCTAssert(g1 == [[1, 2], [3, 4]])
            XCTAssert(g2 == [[1, 2], [3, 4]])
        }
        do {
            let (g1, g2) = pullback(at: a, b, in: { $0 / $1 })(v)
            let data = [1, 0.5, 0.333333343, 0.25].map { CF($0) }
            let g1Expected = array(data, (2, 2))
            let g1sumdiff = sum(g1 - g1Expected).element
            XCTAssert(abs(g1sumdiff.real) <= 1e-6 && g1sumdiff.imaginary == 0)

            let g2Expected = -array(data, (2, 2))
            let g2sumdiff = sum(g2 - g2Expected).element
            XCTAssert(abs(g2sumdiff.real) <= 1e-6 && g2sumdiff.imaginary == 0)
        }
    }

    //--------------------------------------------------------------------------
    // test_subtract
    func test_subtract() {
        let a = array(1..<7, (3, 2))
        let b = array(0..<6, (3, 2))
        let result = a - b
        XCTAssert(result.flat == [1, 1, 1, 1, 1, 1])

        let (g1, g2) = pullback(at: a, b, in: { $0 - $1 })(ones(like: a))
        XCTAssert(g1.flat == [1, 1, 1, 1, 1, 1])
        XCTAssert(g2.flat == [-1, -1, -1, -1, -1, -1])
    }

    //--------------------------------------------------------------------------
    // test_subtractScalar
    func test_subtractScalar() {
        let a = array(1...6, (3, 2))
        let result = a - 1
        XCTAssert(result == [[0, 1], [2, 3], [4, 5]])

        let result2 = 1 - a
        XCTAssert(result2 == [[0, -1], [-2, -3], [-4, -5]])
    }

    //--------------------------------------------------------------------------
    // test_subtractVector
    func test_subtractVector() {
        let a = array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])
        let col = repeating(array(0...2, (3, 1)), (3, 2))

        let result = a - col
        let expected: [[Float]] = [
            [1, 2],
            [2, 3],
            [3, 4]
        ]
        XCTAssert(result == expected)

        let result2 = col - a
        let expected2: [[Float]] = [
            [-1, -2],
            [-2, -3],
            [-3, -4]
        ]
        XCTAssert(result2 == expected2)
    }

    //--------------------------------------------------------------------------
    // test_subtractAndAssign
    func test_subtractAndAssign() {
        var a = array(1...6, (3, 2))
        a -= 1
        XCTAssert(a == [[0, 1], [2, 3], [4, 5]])
    }

    //--------------------------------------------------------------------------
    // test_mul
    func test_mul() {
        let a = array(0..<6, (3, 2))
        let b = array(0..<6, (3, 2))
        let result = a * b
        XCTAssert(result == [[0, 1], [4, 9], [16, 25]])

        let (g1, g2) = pullback(at: a, b, in: { $0 * $1 })(ones(like: a))
        XCTAssert(g1 == [[0, 1], [2, 3], [4, 5]])
        XCTAssert(g2 == [[0, 1], [2, 3], [4, 5]])
    }

    //--------------------------------------------------------------------------
    // test_mulScalar
    func test_mulScalar() {
        let a = array(1...6, (3, 2))
        let result = a * 2
        XCTAssert(result == [[2, 4], [6, 8], [10, 12]])
    }

    //--------------------------------------------------------------------------
    // test_mulAndAssign
    func test_mulAndAssign() {
        var a = array(1...6, (3, 2))
        a *= 2
        XCTAssert(a == [[2, 4], [6, 8], [10, 12]])
    }

    //--------------------------------------------------------------------------
    // test_div
    func test_div() {
        let a = array([[1, 4], [9, 16], [25, 36]])
        let b = array(1...6, (3, 2))
        let result = a / b
        XCTAssert(result == [[1, 2], [3, 4], [5, 6]])

        do {
            let (g1, g2) = pullback(at: a, b, in: { $0 / $1 })(ones(like: a))
            let g1Expected = array([[1, 0.5], [0.3333333, 0.25], [0.2, 0.1666666]])
            XCTAssert(abssum(g1 - g1Expected).element <= 1e-6)
            XCTAssert(g2.array == [[-1, -1], [-1, -1], [-1, -1]])
        }
    }

    //--------------------------------------------------------------------------
    // test_divScalar
    func test_divScalar() {
        let a = array(1...6, (3, 2))
        let result = a / 2
        XCTAssert(result == [[0.5, 1], [1.5, 2], [2.5, 3]])
    }

    //--------------------------------------------------------------------------
    // test_divAndAssign
    func test_divAndAssign() {
        var a = array(1...6, (3, 2))
        a /= 2
        XCTAssert(a == [[0.5, 1], [1.5, 2], [2.5, 3]])
    }
}
