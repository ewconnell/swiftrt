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

class test_Spatial: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_poolAverage", test_poolAverage),
    ("test_poolAveragePadding", test_poolAveragePadding),
    ("test_poolMax", test_poolMax),
  ]

  //--------------------------------------------------------------------------
  func test_poolAverage() {
    #if canImport(SwiftRTCuda)
      do {
        let a = array([
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
        ])

        // same
        let same = pool(x: a, windowSize: 3, strides: 1, padding: .same, mode: .average)
        XCTAssert(a.shape == same.shape)
        XCTAssert(
          same == [
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
          ])

        // default is strides 1 padding 0
        let valid = pool(x: a, windowSize: 3, mode: .average)
        XCTAssert(valid == [[4.0]])

        // using a configuration
        let config = PoolingConfiguration(
          x: a, windowSize: (3, 3), strides: (1, 1), padding: .valid, mode: .average)
        var out = config.createOutput()
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[4.0]])
      }

      do {
        let a = array(0..<25, shape: (5, 5))

        // same
        let same = pool(x: a, windowSize: (5, 5), padding: .same, mode: .average)
        let expsame = array([
          [6.0, 6.5, 7.0, 7.5, 8.0],
          [8.5, 9.0, 9.5, 10.0, 10.5],
          [11.0, 11.5, 12.0, 12.5, 13.0],
          [13.5, 14.0, 14.5, 15.0, 15.5],
          [16.0, 16.5, 17.0, 17.5, 18.0],
        ])
        XCTAssert(elementsAlmostEqual(same, expsame, tolerance: 0.001).all().element)

        // valid
        let valid = pool(x: a, windowSize: 5, padding: 0, mode: .average)
        print(valid)
        XCTAssert(valid == [[12.0]])

        // using a configuration
        let config = PoolingConfiguration(
          x: a, windowSize: (5, 5), strides: (1, 1), padding: .valid, mode: .average)
        var out = config.createOutput()
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[12.0]])
      }
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolAveragePadding() {
    #if canImport(SwiftRTCuda)
      do {
        let a = array([
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
        ])

        // same
        let same = pool(
          x: a, windowSize: (3, 3), strides: (1, 1), padding: .same, mode: .averagePadding)
        print(same)
        XCTAssert(a.shape == same.shape)
        XCTAssert(
          same == [
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
          ])

        // valid
        let valid = pool(
          x: a, windowSize: (3, 3), strides: (1, 1), padding: .valid, mode: .averagePadding)
        XCTAssert(valid == [[4.0]])

        // using a configuration
        let config = PoolingConfiguration(
          x: a, windowSize: (3, 3), strides: (1, 1), padding: .valid, mode: .averagePadding)
        var out = config.createOutput()
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[4.0]])
      }

      do {
        let a = array(0..<25, shape: (5, 5))

        // same
        let same = pool(
          x: a, windowSize: (5, 5), strides: (1, 1), padding: .same, mode: .averagePadding)
        let expsame = array([
          [6.0, 6.5, 7.0, 7.5, 8.0],
          [8.5, 9.0, 9.5, 10.0, 10.5],
          [11.0, 11.5, 12.0, 12.5, 13.0],
          [13.5, 14.0, 14.5, 15.0, 15.5],
          [16.0, 16.5, 17.0, 17.5, 18.0],
        ])
        XCTAssert(elementsAlmostEqual(same, expsame, tolerance: 0.001).all().element)

        // valid
        let valid = pool(
          x: a, windowSize: (5, 5), strides: (1, 1), padding: .valid, mode: .averagePadding)
        print(valid)
        XCTAssert(valid == [[12.0]])

        // using a configuration
        let config = PoolingConfiguration(
          x: a, windowSize: (5, 5), strides: (1, 1), padding: .valid, mode: .averagePadding)
        var out = config.createOutput()
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[12.0]])
      }
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolMax() {
    #if canImport(SwiftRTCuda)
      let a = array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ])

      // same
      let same = pool(x: a, windowSize: (3, 3), strides: (1, 1), padding: .same, mode: .max)
      XCTAssert(
        same == [
          [4.0, 5.0, 5.0],
          [7.0, 8.0, 8.0],
          [7.0, 8.0, 8.0],
        ])

      // valid
      let valid = pool(x: a, windowSize: (3, 3), strides: (1, 1), padding: .valid, mode: .max)
      XCTAssert(valid == [[8.0]])
    #endif
  }
}
