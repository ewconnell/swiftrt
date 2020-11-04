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

class test_Casting: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_cast2Int", test_cast2Int),
    ("test_cast2Float", test_cast2Float),
    ("test_cast2Bool", test_cast2Bool),
  ]

  //--------------------------------------------------------------------------
  func test_cast2Int() {
    do {
      let f = array([[0, 1], [2, 3], [4, 5]])
      XCTAssert(TensorR2<Int32>(f) == [[0, 1], [2, 3], [4, 5]])

      let f16 = array([[0, 1], [2, 3], [4, 5]], type: Float16.self)
      XCTAssert(TensorR2<Int32>(f16) == [[0, 1], [2, 3], [4, 5]])

      let b = array([true, false, true, false, false])
      XCTAssert(TensorR1<Int32>(b) == [1, 0, 1, 0, 0])
    }
  }

  //--------------------------------------------------------------------------
  func test_cast2Float() {
    do {
      let i = array(0..<6, (3, 2), type: Int32.self)
      XCTAssert(Tensor2(i) == [[0, 1], [2, 3], [4, 5]])
    }

    do {
      let b = array([true, false, true, false, false])
      XCTAssert(Tensor1(b) == [1, 0, 1, 0, 0])
    }
  }

  //--------------------------------------------------------------------------
  func test_cast2Bool() {
    do {
      let f = array([1, 0, 1, 0, 0])
      XCTAssert(TensorR1<Bool>(f) == [true, false, true, false, false])
    }
  }

  //--------------------------------------------------------------------------
  func test_cast2Complex() {
    // TODO
    // typealias CF16 = Complex<Float16>
    // typealias CF = Complex<Float>
    // let f = array(0..<4)
    // let cf16 = array([CF16(0), CF16(1), CF16(2)])
    // let cf = array([CF(0), CF(1), CF(2)])

    // XCTAssert(TensorR1<CF16>(f) == cf16)

    // XCTAssert(TensorR1<Complex<Float>>(cf16) == cf)
  }

}
