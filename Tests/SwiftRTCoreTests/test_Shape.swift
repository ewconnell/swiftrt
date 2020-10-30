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
import SwiftRT
import XCTest
import _Differentiation

class test_Shape: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_reshapeOrderRowCol", test_reshapeOrderRowCol),
    ("test_fillRangeColumnMajor", test_fillRangeColumnMajor),
    ("test_reshape", test_reshape),
    ("test_reshapeOrderRowTC32x8", test_reshapeOrderRowTC32x8),
    ("test_reshapeOrderRowTC32x32", test_reshapeOrderRowTC32x32),
    ("test_expanding", test_expanding),
    ("test_expandMutate", test_expandMutate),
    ("test_repeatExpandTranspose", test_repeatExpandTranspose),
    ("test_stridePermutation", test_stridePermutation),
    ("test_squeeze", test_squeeze),
    ("test_stack", test_stack),
    ("test_stackingGradients", test_stackingGradients),
    ("test_stackingExpression", test_stackingExpression),
    ("test_perfTensor1", test_perfTensor1),
    ("test_perfTensor2", test_perfTensor2),
    ("test_perfRepeatedTensor3", test_perfRepeatedTensor3),
    ("test_perfTensor3", test_perfTensor3),
    ("test_perfTensor4", test_perfTensor4),
    ("test_perfTensor5", test_perfTensor5),
    ("test_initEmpty", test_initEmpty),
    ("test_initRepeating", test_initRepeating),
    ("test_initSingle", test_initSingle),
    ("test_contiguousViews", test_contiguousViews),
    ("test_transposed", test_transposed),
    ("testTransposedPullback", testTransposedPullback),
  ]

  //--------------------------------------------------------------------------
  func test_fillRangeColumnMajor() { testEachDevice(fillRangeColumnMajor) }

  func fillRangeColumnMajor() {
    let a = array(from: Float(0), to: Float(3), (2, 3), order: .row)
    let b = array(from: Float(0), to: Float(3), (2, 3), order: .col)
    XCTAssert(a.array == b.array)

    typealias CF = Complex<Float>
    let c = array(from: CF(0), to: CF(3), (2, 3), order: .row)
    let d = array(from: CF(0), to: CF(3), (2, 3), order: .col)
    XCTAssert(c.array == d.array)
  }

  //--------------------------------------------------------------------------
  func test_reshapeOrderRowCol() { testEachDevice(reshapeOrderRowCol) }

  func reshapeOrderRowCol() {
    let a = array([[0, 1, 2], [3, 4, 5]])
    XCTAssert(Array(a.read()) == [0, 1, 2, 3, 4, 5])

    let b = reshape(a, (2, 3), order: .col)
    XCTAssert(Array(b.read()) == [0, 3, 1, 4, 2, 5])
    XCTAssert(b == [[0, 1, 2], [3, 4, 5]])

    let c = array([[0, 3, 1], [4, 2, 5]], order: .col)
    XCTAssert(Array(c.buffer) == [0, 3, 1, 4, 2, 5])

    let d = reshape(c, (2, 3))
    XCTAssert(d == [[0, 1, 2], [3, 4, 5]])
    XCTAssert(Array(d.buffer) == [0, 1, 2, 3, 4, 5])
  }

  //--------------------------------------------------------------------------
  func test_reshape() {
    let a3 = array(0..<12, (2, 3, 2))

    // R3 -> R2
    let a2 = reshape(a3, (2, -1))
    XCTAssert(a2.shape == [2, 6])
    XCTAssert(a2 == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

    // R3 -> R1
    let a1 = reshape(a3, -1)
    XCTAssert(a1 == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    // R1 -> R2
    let b2 = reshape(a1, (2, -1))
    XCTAssert(b2 == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

    // R1 -> R3
    let b3 = reshape(a1, (2, 2, 3))
    XCTAssert(b3 == [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    let input = ones((2, 4))
    let reshapedPullback = pullback(at: input) {
      reshape($0, (2, 2, 2))
    }
    let reshaped = ones((2, 2, 2))
    XCTAssertEqual(input, reshapedPullback(reshaped))
  }

  //--------------------------------------------------------------------------
  func test_repeatExpandTranspose() {
    let a = repeating(array(0...3, (4, 1, 1)), (4, 3, 4))
    let b = repeating(expand(dims: array(0...3), axes: (0, 1)), (3, 4, 4))
      .transposed(permutatedBy: Shape3(2, 0, 1))
    XCTAssert(a == b)
  }

  //--------------------------------------------------------------------------
  func test_stridePermutation() {
    let maxj = 20
    let maxi = 20
    let maxk = 8

    let vec = array([
      0, 0, 0, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 0, 0, 0,
    ])
    let ts = repeating(expand(dims: vec, axes: (0, 1, 3)), (2, maxj, maxi, maxk))
    XCTAssert(ts[0, ..., ..., 0].count == (maxj * maxi))

    let s = ts[0, 0..<maxj, 0..<maxi, 0..<maxk]
    let tp = s.transposed(permutatedBy: [3, 1, 2, 0])
    let test = squeeze(tp, axis: 3)

    //Below breaks in DiscreteStorage assert
    XCTAssert(
      test[7] == [
        [
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
          [0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0],
        ]
      ])
  }

  //--------------------------------------------------------------------------
  // bug repro from user
  func test_expandMutate() {
    let maxi = 8
    let maxj = 8
    var drag = repeating(0, (2, maxj + 2, maxi + 2))
    let idxoffset1 = 6
    let idxoffset2 = 2
    let data = repeating(
      array([1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0], (1, maxi + 1)),
      ((idxoffset1 - idxoffset2), maxi + 1))
    drag[1, (maxj - idxoffset1)..<(maxj - idxoffset2), 0...maxi] = expand(dims: data, axis: 0)
    drag[0, (maxj - idxoffset1)..<(maxj - idxoffset2), 0...maxi] =
      drag[1, (maxj - idxoffset1)..<(maxj - idxoffset2), 0...maxi]
    XCTAssert(
      drag == [
        [
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [1.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 1.5, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  func test_reshapeOrderRowTC32x8() {
    let a = array([[0, 1, 2], [3, 4, 5]])
    XCTAssert(a.flatArray == [0, 1, 2, 3, 4, 5])

    let b = reshape(a, (2, 3), order: .col)
    XCTAssert(b == [[0, 1, 2], [3, 4, 5]])
    XCTAssert(b.flatArray == [0, 3, 1, 4, 2, 5])

    let c = array([[0, 3, 1], [4, 2, 5]], order: .col)
    XCTAssert(c.flatArray == [0, 3, 1, 4, 2, 5])

    let d = reshape(c, (2, 3))
    XCTAssert(d == [[0, 1, 2], [3, 4, 5]])
    XCTAssert(d.flatArray == [0, 1, 2, 3, 4, 5])
  }

  //--------------------------------------------------------------------------
  func test_reshapeOrderRowTC32x32() {
    let a = array([[0, 1, 2], [3, 4, 5]])
    XCTAssert(a.flatArray == [0, 1, 2, 3, 4, 5])

    let b = reshape(a, (2, 3), order: .col)
    XCTAssert(b == [[0, 1, 2], [3, 4, 5]])
    XCTAssert(b.flatArray == [0, 3, 1, 4, 2, 5])

    let c = array([[0, 3, 1], [4, 2, 5]], order: .col)
    XCTAssert(c.flatArray == [0, 3, 1, 4, 2, 5])

    let d = reshape(c, (2, 3))
    XCTAssert(d == [[0, 1, 2], [3, 4, 5]])
    XCTAssert(d.flatArray == [0, 1, 2, 3, 4, 5])
  }

  //--------------------------------------------------------------------------
  func test_expanding() {
    let a = array(0..<4)
    let b = expand(dims: a, axis: 0)
    XCTAssert(b.shape == [1, 4])
    XCTAssert(b.strides == [4, 1])
    XCTAssert(b == [[0, 1, 2, 3]])

    let c = expand(dims: b, axes: (3, 0))
    XCTAssert(c.shape == [1, 1, 4, 1])
    XCTAssert(c.strides == [4, 4, 1, 1])
    XCTAssert(c == [[[[0], [1], [2], [3]]]])

    // test derivatives
    func f1(a: Tensor1) -> Tensor2 { expand(dims: a, axis: 0).squared() }
    func f2(a: Tensor1) -> Tensor2 { expand(dims: a.squared(), axis: 0) }
    XCTAssert(pullback(at: array([3, 5]), in: f1)(array([[1, 1]])) == [6, 10])
    XCTAssert(pullback(at: array([3, 5]), in: f2)(array([[1, 1]])) == [6, 10])
  }

  //--------------------------------------------------------------------------
  func test_squeeze() {
    let a = array(0..<24, (2, 3, 4))

    let sumCols = a.sum(axes: 2)
    XCTAssert(sumCols.shape == [2, 3, 1])
    let b = squeeze(sumCols, axis: -1)
    XCTAssert(
      b == [
        [6.0, 22.0, 38.0],
        [54.0, 70.0, 86.0],
      ])

    let sumRows = a.sum(axes: 1)
    XCTAssert(sumRows.shape == [2, 1, 4])
    let c = squeeze(sumRows, axis: 1)
    XCTAssert(
      c == [
        [12.0, 15.0, 18.0, 21.0],
        [48.0, 51.0, 54.0, 57.0],
      ])

    // test negative axes
    let d = squeeze(sumRows, axis: -2)
    XCTAssert(
      d == [
        [12.0, 15.0, 18.0, 21.0],
        [48.0, 51.0, 54.0, 57.0],
      ])

    // test derivatives
    func f1(a: Tensor2) -> Tensor1 { squeeze(a, axis: 0).squared() }
    func f2(a: Tensor2) -> Tensor1 { squeeze(a.squared(), axis: 0) }
    XCTAssert(pullback(at: array([[3, 5]]), in: f1)(array([1, 1])) == [[6, 10]])
    XCTAssert(pullback(at: array([[3, 5]]), in: f2)(array([1, 1])) == [[6, 10]])
  }

  //--------------------------------------------------------------------------
  func test_stack() {
    let a = array(0..<6, (2, 3))
    let b = array(6..<12, (2, 3))

    let v0 = stack(a, b)
    XCTAssert(v0 == [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])

    let v1 = stack(a, b, axis: 1)
    XCTAssert(
      v1 == [
        [[0, 1, 2], [6, 7, 8]],
        [[3, 4, 5], [9, 10, 11]],
      ])

    let v2 = stack(a, b, axis: 2)
    XCTAssert(
      v2 == [
        [
          [0, 6],
          [1, 7],
          [2, 8],
        ],

        [
          [3, 9],
          [4, 10],
          [5, 11],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  func test_stackingGradients() {
    let a1 = array([1, 2, 3, 4, 5])
    let b1 = array([6, 7, 8, 9, 10])
    let a2 = ones((5))
    let b2 = ones((5))

    //        let c = stack(a1 * a2, b1 * b2, axis: -1).sum().element

    let grads = gradient(at: a2, b2) { a, b in
      stack(a1 * a, b1 * b, axis: -1).sum().element
    }
    XCTAssertEqual(a1, grads.0)
    XCTAssertEqual(b1, grads.1)
  }

  //--------------------------------------------------------------------------
  func test_stackingExpression() {
    let i = 3
    let j = 3
    let maxK: Float = 16
    let k1 = array(0..<30, (5, 6))

    let mask =
      squeeze(
        stack([
          k1[0...j, 1...i],
          k1[0...j, 2...i + 1],
          k1[1...j + 1, 1...i],
          k1[1...j + 1, 2...i + 1],
        ]).max(axes: 0), axis: 0) .<= maxK

    XCTAssert(
      mask.array == [
        [true, true, true],
        [true, true, true],
        [false, false, false],
        [false, false, false],
      ])
  }

  //--------------------------------------------------------------------------
  func test_perfTensor1() {
    #if !DEBUG
      let a = ones(1024 * 1024)
      var count: DType = 0
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_perfTensor2() {
    #if !DEBUG
      let a = ones((1024, 1024))
      var count: DType = 0

      // 0.001s
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_perfRepeatedTensor3() {
    #if !DEBUG
      let a = repeating(1, (64, 128, 128))
      var count: DType = 0
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_perfTensor3() {
    #if !DEBUG
      let a = ones((64, 128, 128))
      var count: DType = 0
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_perfTensor4() {
    #if !DEBUG
      let a = ones((2, 32, 128, 128))
      var count: DType = 0
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_perfTensor5() {
    #if !DEBUG
      let a = ones((2, 2, 16, 128, 128))
      var count: DType = 0
      self.measure {
        count = a.sum().element
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_initEmpty() {
    #if !DEBUG
      var count: DType = 0
      self.measure {
        for _ in 0..<100000 {
          let a = Tensor2(shape: Shape2(2, 5))
          count = a.first
        }
      }
      XCTAssert(count != 3.1415926)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_initRepeating() {
    #if !DEBUG
      var count: DType = 0
      self.measure {
        for _ in 0..<100000 {
          let a = Tensor1(repeating: 1, to: Shape1(1))
          count += a.first
        }
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_initSingle() {
    #if !DEBUG
      var count: DType = 0
      self.measure {
        for _ in 0..<100000 {
          let a = Tensor1(1)
          count += a.first
        }
      }
      XCTAssert(count > 0)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_contiguousViews() {
    // vector views are always sequential
    let v = array(0..<6)
    let subv = v[1...2]
    XCTAssert(subv.isContiguous)

    // a batch of rows are sequential
    let m = empty((4, 5))
    let mrows = m[1...2, ...]
    XCTAssert(mrows.isContiguous)

    // a batch of columns are not sequential
    let m1 = empty((4, 5))
    let mcols = m1[..., 1...2]
    XCTAssert(!mcols.isContiguous)
  }

  //--------------------------------------------------------------------------
  func test_transposed() {
    let m = array(0..<9, (3, 3))
    XCTAssert(m.t == [[0, 3, 6], [1, 4, 7], [2, 5, 8]])

    let a = array(0..<24, (2, 3, 4))
    let transA = a.transposed(permutatedBy: [2, 1, 0])
    XCTAssert(
      transA == [
        [
          [0.0, 12.0],
          [4.0, 16.0],
          [8.0, 20.0],
        ],

        [
          [1.0, 13.0],
          [5.0, 17.0],
          [9.0, 21.0],
        ],

        [
          [2.0, 14.0],
          [6.0, 18.0],
          [10.0, 22.0],
        ],

        [
          [3.0, 15.0],
          [7.0, 19.0],
          [11.0, 23.0],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  func testTransposedPullback() {
    let input = ones((2, 3))
    let transposed = ones((3, 2))
    let transposedPullback = pullback(at: input) { $0.t }
    let transposedPermutationsPullback = pullback(at: input) {
      $0.transposed(permutatedBy: [1, 0])
    }
    let transposedVariadicsPullback = pullback(at: input) {
      $0.transposed(permutatedBy: [1, 0])
    }

    XCTAssertEqual(input, transposedPullback(transposed))
    XCTAssertEqual(input, transposedPermutationsPullback(transposed))
    XCTAssertEqual(input, transposedVariadicsPullback(transposed))
  }
}
