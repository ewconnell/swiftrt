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

class test_PackedElements: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_castingBool1_UInt1", test_castingBool1_UInt1),
    ("test_Bool1", test_Bool1),
    ("test_UInt1", test_UInt1),
    ("test_UInt4", test_UInt4),
  ]

  //--------------------------------------------------------------------------
  func test_castingBool1_UInt1() {
    do {
      let a = array([false, true, false, true], type: Bool1.self)
      XCTAssert(a.storage.countOf(type: Bool1.Stored.self) == 1)
      XCTAssert(a == [false, true, false, true])

      let b = TensorR1<UInt1>(a)
      XCTAssert(b.storage.countOf(type: UInt1.Stored.self) == 1)
      XCTAssert(b == [0, 1, 0, 1])

      let c = TensorR1<Bool1>(b)
      XCTAssert(c.storage.countOf(type: Bool1.Stored.self) == 1)
      XCTAssert(c == a)
    }
  }

  //--------------------------------------------------------------------------
  func test_Bool1() {
    // element wise OR
    do {
      let a = array([false, true, false, true], type: Bool1.self)
      let b = array([true, false, false, false], type: Bool1.self)
      let c = a .|| b
      XCTAssert(c.storage.countOf(type: Bool1.Stored.self) == 1)
      XCTAssert(c == [true, true, false, true])
    }

    // element wise AND cross packing boundary
    do {
      // 10 elements
      let a = array(
        [
          true, true, true, true, false,
          true, false, false, true, true,
        ], type: Bool1.self)
      let b = array(
        [
          false, true, true, true, false,
          true, false, false, false, true,
        ], type: Bool1.self)
      let c = a .&& b
      XCTAssert(c.storage.countOf(type: Bool1.Stored.self) == 2)
      XCTAssert(
        c == [
          false, true, true, true, false,
          true, false, false, false, true,
        ])
    }

    // modify element
    do {
      var a = array(
        [
          false, true, false, true,
          false, true, false, true,
        ], type: Bool1.self)
      a[2] = true
      XCTAssert(a == [false, true, true, true, false, true, false, true])
      a[1] = false
      XCTAssert(a == [false, false, true, true, false, true, false, true])
      a[7] = false
      XCTAssert(a == [false, false, true, true, false, true, false, false])
    }

    // modify range across packing boundaries
    do {
      var a = array(
        [
          [false, false, false, true, false],
          [false, false, true, true, false],
        ], type: Bool1.self)
      XCTAssert(a.storage.countOf(type: Bool1.Stored.self) == 2)
      XCTAssert(
        a == [
          [false, false, false, true, false],
          [false, false, true, true, false],
        ])

      let row = array([true, false, false], shape: (1, 3), type: Bool1.self)
      a[1, 1...3] = row
      XCTAssert(
        a == [
          [false, false, false, true, false],
          [false, true, false, false, false],
        ])
    }
  }

  //--------------------------------------------------------------------------
  func test_UInt1() {
    do {
      let a = array([0, 1, 0, 1], type: UInt1.self)
      let b = array([1, 0, 0, 0], type: UInt1.self)
      let c = a + b
      XCTAssert(c.storage.countOf(type: UInt1.Stored.self) == 1)
      XCTAssert(c == [1, 1, 0, 1])
    }

    // cross packing boundary
    do {
      // 10 elements
      let a = array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0], type: UInt1.self)
      let b = array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], type: UInt1.self)
      let c = a + b
      XCTAssert(c.storage.countOf(type: UInt1.Stored.self) == 2)
      XCTAssert(c == [0, 1, 1, 1, 0, 1, 0, 1, 1, 1])
    }

    // modify element
    do {
      var a = array([0, 1, 0, 1, 0, 1, 0, 1], type: UInt1.self)
      a[2] = 1
      XCTAssert(a == [0, 1, 1, 1, 0, 1, 0, 1])
      a[1] = 0
      XCTAssert(a == [0, 0, 1, 1, 0, 1, 0, 1])
      a[7] = 0
      XCTAssert(a == [0, 0, 1, 1, 0, 1, 0, 0])
    }

    // modify range across packing boundaries
    do {
      var a = array([[0, 0, 0, 1, 0], [0, 0, 1, 1, 0]], type: UInt1.self)
      XCTAssert(a.storage.countOf(type: UInt1.Stored.self) == 2)
      XCTAssert(
        a == [
          [0, 0, 0, 1, 0],
          [0, 0, 1, 1, 0],
        ])

      let row = array([1, 0, 0], shape: (1, 3), type: UInt1.self)
      a[1, 1...3] = row
      XCTAssert(
        a == [
          [0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0],
        ])
    }
  }

  //--------------------------------------------------------------------------
  func test_UInt4() {
    // even packing alignment
    do {
      let a = array(0..<4, type: UInt4.self)
      let b = array(1..<5, type: UInt4.self)
      let c = a + b
      XCTAssert(c.storage.countOf(type: UInt4.Stored.self) == 2)
      XCTAssert(c == [1, 3, 5, 7])
    }

    // odd packing alignment
    do {
      let a = array(0..<5, type: UInt4.self)
      let b = array(1..<6, type: UInt4.self)
      let c = a + b
      XCTAssert(c.storage.countOf(type: UInt4.Stored.self) == 3)
      XCTAssert(c == [1, 3, 5, 7, 9])
    }

    // modify element
    do {
      var a = array(0..<4, type: UInt4.self)
      a[2] = 7
      XCTAssert(a == [0, 1, 7, 3])
      a[1] = 5
      XCTAssert(a == [0, 5, 7, 3])
    }

    // modify range across packing boundaries
    do {
      var a = array(0..<8, shape: (2, 4), type: UInt4.self)
      XCTAssert(a.storage.countOf(type: UInt4.Stored.self) == 4)
      XCTAssert(
        a == [
          [0, 1, 2, 3],
          [4, 5, 6, 7],
        ])

      let row = array([3, 3], shape: (1, 2), type: UInt4.self)
      XCTAssert(row.storage.countOf(type: UInt4.Stored.self) == 1)
      a[1, 1...2] = row
      XCTAssert(
        a == [
          [0, 1, 2, 3],
          [4, 3, 3, 7],
        ])
    }
  }
}
