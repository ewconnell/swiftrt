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

class test_arraySyntax: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_initSyntax", test_initSyntax),
    ("test_array", test_array),
    ("test_empty", test_empty),
    ("test_emptyLike", test_emptyLike),
    ("test_ones", test_ones),
    ("test_onesLike", test_onesLike),
    ("test_onesView", test_onesView),
    ("test_zeros", test_zeros),
    ("test_zerosLike", test_zerosLike),
    ("test_full", test_full),
    ("test_fullLike", test_fullLike),
    ("test_identity", test_identity),
    ("test_eye", test_eye),
  ]

  //--------------------------------------------------------------------------
  func test_initSyntax() {
    // stored bit pattern
    let _ = array(stored: [Float16(0), Float16(1)])
    let _ = array(stored: [Float16(0), Float16(1)], (1, 2))

    // packed types
    let _ = array([0, 1, 0, 1], type: UInt1.self)
    let _ = array(stored: [0b00001010])

    let _ = array([0, 1, 2, 3], type: UInt4.self)
    let _ = array(stored: [0x10, 0x32])

    // boolean conversion to Element
    let _ = array([true, false], type: UInt1.self)
    let _ = array([true, false], type: UInt4.self)
    let _ = array([true, false], type: UInt8.self)
    let _ = array([true, false], type: Int32.self)
    let _ = array([true, false], type: Float16.self)
    let _ = array([true, false], type: Float.self)
    let _ = array([true, false], type: Double.self)

    // boolean conversion to shaped Element
    let _ = array([true, false], (1, 2), type: UInt1.self)
    let _ = array([true, false], (1, 2), type: UInt4.self)
    let _ = array([true, false], (1, 2), type: UInt8.self)
    let _ = array([true, false], (1, 2), type: Int32.self)
    let _ = array([true, false], (1, 2), type: Float16.self)
    let _ = array([true, false], (1, 2), type: Float.self)
    let _ = array([true, false], (1, 2), type: Double.self)

    // implicit vectors
    let _ = array([true, false])
    let _ = array([0, 1, 2])
    let _ = array([Float](arrayLiteral: 0, 1, 2))
    let _ = array([0.0, 1.5, 2.5])
    let _ = array([RGBA<UInt8>(0, 127, 255, 255), RGBA<UInt8>(63, 127, 191, 255)])
    let _ = array([RGBA<Float>(0, 0.5, 1, 1), RGBA<Float>(0.25, 0.5, 0.75, 1)])

    // implicit shaped
    let _ = array([true, false], (1, 2))
    let _ = array([RGBA<UInt8>(0, 127, 255, 255), RGBA<UInt8>(63, 127, 191, 255)], (1, 2))
    let _ = array([RGBA<Float>(0, 0.5, 1, 1), RGBA<Float>(0.25, 0.5, 0.75, 1)], (1, 2))

    // integer conversions to Element
    let _ = array([0, 1, 2], type: Bool.self)
    let _ = array([0, 1, 2], type: UInt8.self)
    let _ = array([0, 1, 2], type: Int32.self)
    let _ = array([0, 1, 2], type: Float.self)
    let _ = array([0, 1, 2], type: Double.self)

    // floating conversions to Element
    let _ = array([0.0, 1, 2], type: Bool.self)
    let _ = array([0.0, 1.5, 2.5], type: UInt8.self)
    let _ = array([0.0, 1.5, 2.5], type: Int32.self)
    let _ = array([0.0, 1.5, 2.5], type: Float.self)
    let _ = array([0.0, 1.5, 2.5], type: Double.self)

    // integer conversions to shaped Element
    let _ = array([0, 1, 2], (1, 3), type: Bool.self)
    let _ = array([0, 1, 2], (1, 3), type: UInt8.self)
    let _ = array([0, 1, 2], (1, 3), type: Int32.self)
    let _ = array([0, 1, 2], (1, 3), type: Float.self)
    let _ = array([0, 1, 2], (1, 3), type: Double.self)

    // floating conversions to shaped Element
    let _ = array([0.0, 1, 2], (1, 3), type: Bool.self)
    let _ = array([0.0, 1.5, 2.5], (1, 3), type: UInt8.self)
    let _ = array([0.0, 1.5, 2.5], (1, 3), type: Int32.self)
    let _ = array([0.0, 1.5, 2.5], (1, 3), type: Float.self)
    let _ = array([0.0, 1.5, 2.5], (1, 3), type: Double.self)
  }

  //--------------------------------------------------------------------------
  // test_array
  func test_array() {
    // Rank1
    let b: [Int8] = [0, 1, 2]
    let _ = array(b)
    let _ = array(b, type: Int32.self)
    let _ = array(b, type: Double.self)

    let dt: [DType] = [1.5, 2.5, 3.5]
    let _ = array(dt)
    let _ = array(dt, type: Int32.self)
    let _ = array(dt, type: Double.self)

    let d: [Double] = [1.5, 2.5, 3.5]
    let _ = array(d)
    let _ = array(d, type: Int32.self)

    let _ = array([[0, 1, 2], [3, 4, 5]])
    let _ = array([0, 1, 2, 3, 4, 5], (2, 3))
    let _ = array(0..<6, (2, 3))
  }

  //--------------------------------------------------------------------------
  // test_empty
  func test_empty() {
    // T0
    let _ = empty()
    let _ = empty(type: Int32.self)

    // T1
    let _ = empty(3)
    let _ = empty(3, order: .F)
    let _ = empty(3, type: Int32.self)
    let _ = empty(3, type: Int32.self, order: .F)

    // T2
    let _ = empty((2, 3))
    let _ = empty((2, 3), order: .F)
    let _ = empty((2, 3), type: Int32.self)
    let _ = empty((2, 3), type: Int32.self, order: .F)
  }

  //--------------------------------------------------------------------------
  // test_emptyLike
  func test_emptyLike() {
    let proto = empty((2, 3))

    let _ = empty(like: proto)
    let _ = empty(like: proto, shape: (6))
    let _ = empty(like: proto, shape: (1, 2, 3))
    let _ = empty(like: proto, order: .F)
    let _ = empty(like: proto, order: .F, shape: (1, 2, 3))
    let _ = empty(like: proto, type: Int32.self)
    let _ = empty(like: proto, type: Int32.self, shape: (1, 2, 3))
    let _ = empty(like: proto, type: Int32.self, order: .F, shape: (1, 2, 3))
  }

  //--------------------------------------------------------------------------
  // test_ones
  func test_ones() {
    // T0
    let _ = one()
    let _ = one(type: Int32.self)

    // T1
    let _ = ones(3)
    let _ = ones(3, order: .F)
    let _ = ones(3, type: Int32.self)
    let _ = ones(3, type: Int32.self, order: .F)

    // T2
    let _ = ones((2, 3))
    let _ = ones((2, 3), order: .F)
    let _ = ones((2, 3), type: Int32.self)
    let _ = ones((2, 3), type: Int32.self, order: .F)
  }

  //--------------------------------------------------------------------------
  // test_onesLike
  func test_onesLike() {
    let proto = ones((2, 3))

    let _ = ones(like: proto)
    let _ = ones(like: proto, shape: (6))
    let _ = ones(like: proto, shape: (1, 2, 3))
    let _ = ones(like: proto, order: .F)
    let _ = ones(like: proto, order: .F, shape: (1, 2, 3))
    let _ = ones(like: proto, type: Int32.self)
    let _ = ones(like: proto, type: Int32.self, shape: (1, 2, 3))
    let _ = ones(like: proto, type: Int32.self, order: .F, shape: (1, 2, 3))
  }

  //--------------------------------------------------------------------------
  // test_onesView
  func test_onesView() {
    let t1 = ones((4, 3), type: Int32.self)
    let view = t1[1...2, ...]
    XCTAssert(view == [[1, 1, 1], [1, 1, 1]])
  }

  //--------------------------------------------------------------------------
  // test_zeros
  func test_zeros() {
    // T0
    let _ = zero()
    let _ = zero(type: Int32.self)

    // T1
    let _ = zeros(3)
    let _ = zeros(3, order: .F)
    let _ = zeros(3, type: Int32.self)
    let _ = zeros(3, type: Int32.self, order: .F)

    // T2
    let _ = zeros((2, 3))
    let _ = zeros((2, 3), order: .F)
    let _ = zeros((2, 3), type: Int32.self)
    let _ = zeros((2, 3), type: Int32.self, order: .F)
  }

  //--------------------------------------------------------------------------
  // test_zerosLike
  func test_zerosLike() {
    let proto = zeros((2, 3))

    let _ = zeros(like: proto)
    let _ = zeros(like: proto, shape: (6))
    let _ = zeros(like: proto, shape: (1, 2, 3))
    let _ = zeros(like: proto, order: .F)
    let _ = zeros(like: proto, order: .F, shape: (1, 2, 3))
    let _ = zeros(like: proto, type: Int32.self)
    let _ = zeros(like: proto, type: Int32.self, shape: (1, 2, 3))
    let _ = zeros(like: proto, type: Int32.self, order: .F, shape: (1, 2, 3))
  }

  //--------------------------------------------------------------------------
  // test_full
  func test_full() {
    // T0
    let _ = full(42)
    let _ = full(42, type: Int32.self)

    // T1
    let _ = full(3)
    let _ = full(3, 42, order: .F)
    let _ = full(3, 42, type: Int32.self)
    let _ = full(3, 42, type: Int32.self, order: .F)

    // T2
    let _ = full((2, 3), 42)
    let _ = full((2, 3), 42, order: .F)
    let _ = full((2, 3), 42, type: Int32.self)
    let _ = full((2, 3), 42, type: Int32.self, order: .F)
  }

  //--------------------------------------------------------------------------
  // test_fullLike
  func test_fullLike() {
    let proto = empty((2, 3))

    let _ = full(like: proto, 42)
    let _ = full(like: proto, 42, shape: (6))
    let _ = full(like: proto, 42, shape: (1, 2, 3))
    let _ = full(like: proto, 42, order: .F)
    let _ = full(like: proto, 42, order: .F, shape: (1, 2, 3))
    let _ = full(like: proto, 42, type: Int32.self)
    let _ = full(like: proto, 42, type: Int32.self, shape: (1, 2, 3))
    let _ = full(like: proto, 42, type: Int32.self, order: .F, shape: (1, 2, 3))
  }

  //--------------------------------------------------------------------------
  // test_identity
  func test_identity() {
    //        let _ = identity(3)
    //        let _ = identity(3, order: .F)
    //        let _ = identity(3, type: Int.self)
    //        let _ = identity(3, type: Int.self, order: .F)
  }

  //--------------------------------------------------------------------------
  // test_eye
  func test_eye() {
    // TODO
    //        // verify signature combinations
    //        let _ = eye(2)
    //        let _ = eye(3, k: 1)
    //        let _ = eye(4, 3, k: -1, type: Int.self)
    //        let _ = eye(3, type: Int.self, order: .F)
    //        print(eye(3, k: 0, type: Int.self))
    //        // check plus
    //        XCTAssert(eye(3, k: 1) == [
    //            [0, 1, 0],
    //            [0, 0, 1],
    //            [0, 0, 0],
    //        ])
    //
    //        // check subview plus
    //        XCTAssert(eye(4, k: 1)[..<3, 1...] == [
    //            [0, 1, 0],
    //            [0, 0, 1],
    //            [0, 0, 0],
    //        ])
    //
    //        // check minus
    //        XCTAssert(eye(3, k: -1) == [
    //            [0, 0, 0],
    //            [1, 0, 0],
    //            [0, 1, 0],
    //        ])
    //
    //        // check subview minus
    //        XCTAssert(eye(4, k: -1)[1..., ..<3] == [
    //            [0, 0, 0],
    //            [1, 0, 0],
    //            [0, 1, 0],
    //        ])
  }
}
