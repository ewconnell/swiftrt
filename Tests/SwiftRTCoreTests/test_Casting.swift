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
    ("test_cast2Complex", test_cast2Complex),
  ]

  //--------------------------------------------------------------------------
  func test_cast2Int() {
    let f = array([[0, 1], [2, 3], [4, 5]])
    XCTAssert(cast(f, elementsTo: Int32.self) == [[0, 1], [2, 3], [4, 5]])

    let f16 = array([[0, 1], [2, 3], [4, 5]], type: Float16.self)
    XCTAssert(cast(f16, elementsTo: Int32.self) == [[0, 1], [2, 3], [4, 5]])

    let b = array([true, false, true, false, false])
    XCTAssert(cast(b, elementsTo: Int32.self) == [1, 0, 1, 0, 0])
  }

  //--------------------------------------------------------------------------
  func test_cast2Float() {
    let i = array(0..<6, (3, 2), type: Int32.self)
    XCTAssert(cast(i, elementsTo: Float.self) == [[0, 1], [2, 3], [4, 5]])

    let b = array([true, false, true, false, false])
    XCTAssert(cast(b, elementsTo: Float.self) == [1, 0, 1, 0, 0])
  }

  //--------------------------------------------------------------------------
  func test_cast2Bool() {
    let f = array([1, 0, 1, 0, 0])
    XCTAssert(cast(f, elementsTo: Bool.self) == [true, false, true, false, false])
  }

  //--------------------------------------------------------------------------
  func test_cast2Complex() {
    typealias CF16 = Complex<Float16>
    typealias CF = Complex<Float>
    let f = array(0..<4)
    let cf16 = array([CF16(0), CF16(1), CF16(2), CF16(3)])
    let cf = array([CF(0), CF(1), CF(2), CF(3)])

    XCTAssert(cast(f, elementsTo: CF16.self) == cf16)
    XCTAssert(cast(cf16, elementsTo: CF.self) == cf)
  }
}
