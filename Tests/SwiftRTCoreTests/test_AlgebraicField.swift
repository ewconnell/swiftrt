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

import Foundation
import Numerics
import SwiftRT
import XCTest
import _Differentiation

class test_AlgebraicField: XCTestCase {
  //--------------------------------------------------------------------------
  // support terminal test run
  static var allTests = [
    ("test_minimalAdd", test_minimalAdd),
    ("test_minimalAddVJP", test_minimalAddVJP),

    ("test_perfAdd", test_perfAdd),
    ("test_add", test_add),
    ("test_addRanks", test_addRanks),
    ("test_addStrided", test_addStrided),
    ("test_addFloat16", test_addFloat16),
    ("test_addBFloat16", test_addBFloat16),

    ("test_addInt32", test_addInt32),
    ("test_addInt8", test_addInt8),
    ("test_addUInt8", test_addUInt8),
    ("test_addScalar", test_addScalar),
    ("test_addAndAssign", test_addAndAssign),

    ("test_subtract", test_subtract),
    ("test_subtractScalar", test_subtractScalar),
    ("test_subtractVector", test_subtractVector),
    ("test_subtractAndAssign", test_subtractAndAssign),

    ("test_mul", test_mul),
    ("test_mulScalar", test_mulScalar),
    ("test_mulAndAssign", test_mulAndAssign),

    ("test_div", test_div),
    ("test_divScalar", test_divScalar),
    ("test_divAndAssign", test_divAndAssign),

    ("test_ComplexFloat", test_ComplexFloat),
    ("test_ComplexFloat16", test_ComplexFloat16),

    ("test_queryMatmulProperties", test_queryMatmulProperties),
    ("test_matmul", test_matmul),
    ("test_batchMatmul", test_batchMatmul),
    ("test_leftBatchMatmul", test_leftBatchMatmul),
    ("test_rightBatchMatmul", test_rightBatchMatmul),
  ]

  //--------------------------------------------------------------------------
  func test_addStrided() {
    let a = array(0..<9, shape: (3, 3), type: Float.self)
    let b = a[..., 1] + 1
    XCTAssert(b == [[2], [5], [8]])
  }

  //--------------------------------------------------------------------------
  func test_minimalAdd() {
    let a = array([[0, 1], [2, 3], [4, 5]], name: "a")
    let b = a + 2
    XCTAssert(b == [[2, 3], [4, 5], [6, 7]])
  }

  //--------------------------------------------------------------------------
  func test_minimalAddVJP() {
    let a = array([[0, 1], [2, 3], [4, 5]], name: "a")
    let v = ones(like: a, name: "ones")

    // only wrt lhs
    let g = pullback(at: a, in: { $0 + 2 })(v)
    XCTAssert(g == [[1, 1], [1, 1], [1, 1]])
  }

  //--------------------------------------------------------------------------
  func test_perfAdd() {
    #if !DEBUG
      let r = 300
      let c = 200
      let a = array(0..<(r * c), (r, c), name: "A")
      let b = array(0..<(r * c), (r, c), name: "B")
      var result = empty((3, 2))

      measure {
        for _ in 0..<1000 {
          result = a + b
        }
        currentQueue.waitForCompletion()
      }
      XCTAssert(result.count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  // let testValue3 = repeating(expand(dims: ones(shape: (1024)), axis: 1), shape: (1024, 3)) +
  // repeating(array([1.0, 1.0, 1.0], shape: (1, 3)), shape: (1024, 3))
  func test_addRanks() {
    let N = 20

    let e1 = expand(dims: ones(shape: (N)), axis: 1)
    let r1 = repeating(e1, shape: (N, 3))

    let a2 = array([1.0, 1.0, 1.0], shape: (1, 3))
    let r2 = repeating(a2, shape: (N, 3))

    let t3 = r1 + r2
    let expected = repeating(2, shape: (N, 3))
    XCTAssert(t3 == expected)
  }

  //--------------------------------------------------------------------------
  func test_add() {
    let a = array(0..<6, shape: (3, 2), name: "A")
    let b = array(0..<6, shape: (3, 2), name: "B")
    let aOnes = ones(like: a)

    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])

    // both
    let expected: [[Float]] = [[1, 1], [1, 1], [1, 1]]
    let (g1, g2) = pullback(at: a, b, in: { $0 + $1 })(aOnes)
    XCTAssert(g1 == expected)
    XCTAssert(g2 == expected)

    // lhs
    let glhs = pullback(at: a, in: { $0 + 2 })(aOnes)
    XCTAssert(glhs == expected)

    // rhs
    let grhs = pullback(at: a, in: { 2 + $0 })(aOnes)
    XCTAssert(grhs == expected)
  }

  //--------------------------------------------------------------------------
  func test_addFloat16() {
    let a = array(0..<6, shape: (3, 2), type: Float16.self)
    let b = array(0..<6, shape: (3, 2), type: Float16.self)
    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
  }

  //--------------------------------------------------------------------------
  func test_addBFloat16() {
    let a = array(0..<6, shape: (3, 2), type: BFloat16.self)
    let b = array(0..<6, shape: (3, 2), type: BFloat16.self)
    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
  }

  //--------------------------------------------------------------------------
  func test_addInt32() {
    let a = array(0..<6, shape: (3, 2), type: Int32.self)
    let b = array(0..<6, shape: (3, 2), type: Int32.self)
    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
  }

  //--------------------------------------------------------------------------
  func test_addInt8() {
    let a = array(0..<6, shape: (3, 2), type: Int8.self)
    let b = array(0..<6, shape: (3, 2), type: Int8.self)
    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
  }

  //--------------------------------------------------------------------------
  func test_addUInt8() {
    let a = array(0..<6, shape: (3, 2), type: UInt8.self)
    let b = array(0..<6, shape: (3, 2), type: UInt8.self)
    let result = a + b
    XCTAssert(result == [[0, 2], [4, 6], [8, 10]])
  }

  //--------------------------------------------------------------------------
  func test_addScalar() {
    let a = array(1...6, shape: (3, 2))
    let result = a + 1
    let expected: [[Float]] = [[2, 3], [4, 5], [6, 7]]
    XCTAssert(result == expected)

    let result2 = 1 + a
    XCTAssert(result2 == expected)
  }

  //--------------------------------------------------------------------------
  func test_addAndAssign() {
    var a = array(0...5, shape: (3, 2))
    a += 2
    XCTAssert(a == [[2, 3], [4, 5], [6, 7]])
  }

  //--------------------------------------------------------------------------
  func test_subtract() {
    let a = array(1..<7, shape: (3, 2))
    let b = array(0..<6, shape: (3, 2))
    let result = a - b
    XCTAssert(result.flatArray == [1, 1, 1, 1, 1, 1])

    // both
    let (g1, g2) = pullback(at: a, b, in: { $0 - $1 })(ones(like: a))
    XCTAssert(g1.flatArray == [1, 1, 1, 1, 1, 1])
    XCTAssert(g2.flatArray == [-1, -1, -1, -1, -1, -1])

    // lhs
    let glhs = pullback(at: a, in: { $0 - 2 })(ones(like: a))
    XCTAssert(glhs.flatArray == [1, 1, 1, 1, 1, 1])

    // rhs
    let grhs = pullback(at: a, in: { 2 - $0 })(ones(like: a))
    XCTAssert(grhs.flatArray == [-1, -1, -1, -1, -1, -1])
  }

  //--------------------------------------------------------------------------
  func test_subtractScalar() {
    let a = array(1...6, shape: (3, 2))
    let result = a - 1
    XCTAssert(result == [[0, 1], [2, 3], [4, 5]])

    let result2 = 1 - a
    XCTAssert(result2 == [[0, -1], [-2, -3], [-4, -5]])
  }

  //--------------------------------------------------------------------------
  func test_subtractVector() {
    let a = array([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    let cols = repeating(array(0...2, shape: (3, 1)), shape: (3, 2))
    XCTAssert(
      cols == [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
      ])

    let result = a - cols
    let expected: [[Float]] = [
      [1, 2],
      [2, 3],
      [3, 4],
    ]
    XCTAssert(result == expected)

    let result2 = cols - a
    let expected2: [[Float]] = [
      [-1, -2],
      [-2, -3],
      [-3, -4],
    ]
    XCTAssert(result2 == expected2)
  }

  //--------------------------------------------------------------------------
  func test_subtractAndAssign() {
    var a = array(1...6, shape: (3, 2))
    a -= 1
    XCTAssert(a == [[0, 1], [2, 3], [4, 5]])
  }

  //--------------------------------------------------------------------------
  func test_mul() {
    let a = array([[0.0, 1], [2, 3], [4, 5]])
    let b = array([[0.0, 1], [2, 3], [4, 5]])
    let result = a * b
    XCTAssert(result == [[0, 1], [4, 9], [16, 25]])

    // both
    let (g1, g2) = pullback(at: a, b, in: { $0 * $1 })(ones(like: a))
    XCTAssert(g1 == [[0, 1], [2, 3], [4, 5]])
    XCTAssert(g2 == [[0, 1], [2, 3], [4, 5]])

    // lhs
    let glhs = pullback(at: a, in: { $0 * 2 })(ones(like: a))
    XCTAssert(glhs.flatArray == [2, 2, 2, 2, 2, 2])

    // rhs
    let grhs = pullback(at: a, in: { 2 * $0 })(ones(like: a))
    XCTAssert(grhs.flatArray == [2, 2, 2, 2, 2, 2])
  }

  //--------------------------------------------------------------------------
  func test_mulScalar() {
    let a = array(1...6, shape: (3, 2))
    let result = a * 2
    XCTAssert(result == [[2, 4], [6, 8], [10, 12]])
  }

  //--------------------------------------------------------------------------
  func test_mulAndAssign() {
    var a = array(1...6, shape: (3, 2))
    a *= 2
    XCTAssert(a == [[2, 4], [6, 8], [10, 12]])
  }

  //--------------------------------------------------------------------------
  func test_div() {
    let a = array([[1, 4], [9, 16], [25, 36]])
    let b = array(1...6, shape: (3, 2))
    let result = a / b
    XCTAssert(result == [[1, 2], [3, 4], [5, 6]])

    let (g1, g2) = pullback(at: a, b, in: { $0 / $1 })(ones(like: a))
    let g1Expected = array([[1, 0.5], [0.3333333, 0.25], [0.2, 0.1666666]])
    XCTAssert(abssum(g1 - g1Expected).element <= 1e-6)
    XCTAssert(g2.array == [[-1, -1], [-1, -1], [-1, -1]])

    // lhs
    let glhs = pullback(at: a, in: { $0 / 2 })(ones(like: a))
    XCTAssert(glhs == [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    // rhs
    let grhs = pullback(at: a, in: { 2 / $0 })(ones(like: a))
    XCTAssert(
      grhs == [
        [-2, -0.125],
        [-0.024691358, -0.0078125],
        [-0.0032, -0.0015432099],
      ])
  }

  //--------------------------------------------------------------------------
  func test_divScalar() {
    let a = array(1...6, shape: (3, 2))
    let result = a / 2
    XCTAssert(result == [[0.5, 1], [1.5, 2], [2.5, 3]])
  }

  //--------------------------------------------------------------------------
  func test_divAndAssign() {
    var a = array(1...6, shape: (3, 2))
    a /= 2
    XCTAssert(a == [[0.5, 1], [1.5, 2], [2.5, 3]])
  }

  //--------------------------------------------------------------------------
  func test_ComplexFloat() {
    // we don't do Complex on the gpu yet, so use the cpu
    typealias CF = Complex<Float>
    let data: [CF] = [1, 2, 3, 4]
    let a = array(data, shape: (2, 2))
    let b = array(data, shape: (2, 2))
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
      let g1Expected = array(data, shape: (2, 2))
      let g1sumdiff = sum(g1 - g1Expected).element
      XCTAssert(abs(g1sumdiff.real) <= 1e-6 && g1sumdiff.imaginary == 0)

      let g2Expected = -array(data, shape: (2, 2))
      let g2sumdiff = sum(g2 - g2Expected).element
      XCTAssert(abs(g2sumdiff.real) <= 1e-6 && g2sumdiff.imaginary == 0)
    }
  }

  //--------------------------------------------------------------------------
  func test_ComplexFloat16() {
    // we don't do Complex on the gpu yet, so use the cpu
    typealias CF = Complex<Float16>
    let data: [CF] = [1, 2, 3, 4]
    let a = array(data, shape: (2, 2))
    let b = array(data, shape: (2, 2))
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
      let g1Expected = array(data, shape: (2, 2))
      let g1sumdiff = sum(g1 - g1Expected).element
      XCTAssert(abs(g1sumdiff.real) <= 1e-6 && g1sumdiff.imaginary == 0)

      let g2Expected = -array(data, shape: (2, 2))
      let g2sumdiff = sum(g2 - g2Expected).element
      XCTAssert(abs(g2sumdiff.real) <= 1e-6 && g2sumdiff.imaginary == 0)
    }
  }

  //--------------------------------------------------------------------------
  func test_queryMatmulProperties() {
    // do {
    //     let a = array(0..<6, (3, 2), type: Float16.self)
    //     print(a)
    //     let b = array(0..<8, (2, 4), type: Float16.self)
    //     print(b)
    //     var c = empty((3, 4), type: Float16.self)
    //     let preferences = MatmulPreferences()
    //     print(preferences)

    //     let props = MatmulAlgorithm.query(
    //         a, b, &c,
    //         accumulatorType: .accumulator16F,
    //         scaleType: .real16F,
    //         preferences: preferences,
    //         using: currentQueue)
    //     print(props)
    // }

    // do {
    //     let a = ones((32, 2))
    //     let b = ones((2, 64))
    //     var c = empty((32, 64))
    //     let props = queryMatmulProperties(a, false, b, false, &c)
    //     print(props)
    // }

    // do {
    //     let a = ones((512, 768))
    //     let b = ones((768, 96))
    //     var c = empty((512, 96))
    //     let props = queryMatmulProperties(a, false, b, false, &c)
    //     print(props)
    // }

    // do {
    //     let a = ones((1024, 1024))
    //     let b = ones((1024, 1024))
    //     var c = empty((1024, 1024))
    //     let props = queryMatmulProperties(a, false, b, false, &c)
    //     print(props)
    // }
  }

  //--------------------------------------------------------------------------
  func test_matmul() {
    let a = array([0, 1, 2, 3, 4, 5], shape: (3, 2))
    let b = array([0, 1, 2, 3, 4, 5, 6, 7], shape: (2, 4))
    let c = matmul(a, b)
    XCTAssert(
      c == [
        [4, 5, 6, 7],
        [12, 17, 22, 27],
        [20, 29, 38, 47],
      ])

    // let (g0, g1) = pullback(at: a, b, in: { matmul($0, $1) } )(ones(like: c))
    // XCTAssert(g0 == [[ 6, 22],
    //                  [ 6, 22],
    //                  [ 6, 22]])

    // XCTAssert(g1 == [[6, 6, 6, 6],
    //                  [9, 9, 9, 9]])
  }

  //--------------------------------------------------------------------------
  func test_batchMatmul() {
    //        let a = array(0..<12, (2, 3, 2))
    //        let b = array(0..<16, (2, 2, 4))
    //        let c = matmul(a, b)
    //        XCTAssert(c == [[[  4.0,   5.0,   6.0,   7.0],
    //                         [ 12.0,  17.0,  22.0,  27.0],
    //                         [ 20.0,  29.0,  38.0,  47.0]],
    //
    //                        [[132.0, 145.0, 158.0, 171.0],
    //                         [172.0, 189.0, 206.0, 223.0],
    //                         [212.0, 233.0, 254.0, 275.0]]])
    //
    //        let (g0, g1) = pullback(at: a, b, in: { matmul($0, $1) } )(ones(like: c))
    //        XCTAssert(g0 == [[[ 6.0, 22.0],
    //                          [ 6.0, 22.0],
    //                          [ 6.0, 22.0]],
    //
    //                         [[38.0, 54.0],
    //                          [38.0, 54.0],
    //                          [38.0, 54.0]]])
    //
    //        XCTAssert(g1 == [[[ 6.0,  6.0,  6.0,  6.0],
    //                          [ 9.0,  9.0,  9.0,  9.0]],
    //
    //                         [[24.0, 24.0, 24.0, 24.0],
    //                          [27.0, 27.0, 27.0, 27.0]]])
  }

  //--------------------------------------------------------------------------
  func test_leftBatchMatmul() {
    //        let a = array(0..<12, (2, 3, 2))
    //        let b = array(0..<8, (2, 4))
    //        let c = matmul(a, b)
    //        XCTAssert(c == [[[ 4,  5,  6,  7],
    //                         [12, 17, 22, 27],
    //                         [20, 29, 38, 47]]])
    //
    //        let (g0, g1) = pullback(at: a, b, in: { matmul($0, $1) } )(ones(like: c))
    //        XCTAssert(g0 == [[[ 6.0, 22.0],
    //                          [ 6.0, 22.0],
    //                          [ 6.0, 22.0]],
    //
    //                         [[ 6.0, 22.0],
    //                          [ 6.0, 22.0],
    //                          [ 6.0, 22.0]]])
    //
    //        XCTAssert(g1 == [[30.0, 30.0, 30.0, 30.0],
    //                         [36.0, 36.0, 36.0, 36.0]])
  }

  //--------------------------------------------------------------------------
  func test_rightBatchMatmul() {
    //        let a = array(0..<6, (3, 2))
    //        let b = array(0..<16, (2, 2, 4))
    //        let c = matmul(a, b)
    //        XCTAssert(c == [[[  4.0,   5.0,   6.0,   7.0],
    //                         [ 12.0,  17.0,  22.0,  27.0],
    //                         [ 20.0,  29.0,  38.0,  47.0]],
    //
    //                        [[ 12.0,  13.0,  14.0,  15.0],
    //                         [ 52.0,  57.0,  62.0,  67.0],
    //                         [ 92.0, 101.0, 110.0, 119.0]]])
    //
    //        let (g0, g1) = pullback(at: a, b, in: { matmul($0, $1) } )(ones(like: c))
    //        XCTAssert(g0 == [[44.0, 76.0],
    //                         [44.0, 76.0],
    //                         [44.0, 76.0]])
    //
    //        XCTAssert(g1 == [[[6.0, 6.0, 6.0, 6.0],
    //                          [9.0, 9.0, 9.0, 9.0]],
    //
    //                         [[6.0, 6.0, 6.0, 6.0],
    //                          [9.0, 9.0, 9.0, 9.0]]])
  }
}
