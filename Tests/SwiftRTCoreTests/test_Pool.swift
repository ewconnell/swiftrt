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

class test_Pool: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_poolAverage1D", test_poolAverage1D),
    ("test_poolBatchAverage1D", test_poolBatchAverage1D),
    ("test_poolAverage2D", test_poolAverage2D),
    ("test_poolBatchAverage2D", test_poolBatchAverage2D),
    ("test_poolAverage3D", test_poolAverage3D),
    ("test_poolBatchAverage3D", test_poolBatchAverage3D),
    ("test_poolAveragePadding", test_poolAveragePadding),
    ("test_poolMax", test_poolMax),
    ("test_pool2DPixelAverage", test_pool2DPixelAverage),
    ("test_poolBatch2DPixelAverage", test_poolBatch2DPixelAverage),
    ("test_pool3DPixelAverage", test_pool3DPixelAverage),
    ("test_poolBatch3DPixelAverage", test_poolBatch3DPixelAverage),
  ]

  //--------------------------------------------------------------------------
  func test_pool1DPixelAverage() {
    #if canImport(SwiftRTCuda)
      typealias Pixel = RGBA<UInt8>
      let pixels = array([Pixel(0), Pixel(1), Pixel(2), Pixel(3), Pixel(4), Pixel(5)])
      let avg = pool(x: pixels, windowSize: 3, padding: 1, op: .average)
      print(avg)
      XCTAssert(avg.shape == pixels.shape)
    // XCTAssert(
    //   same == [
    //     [2.0, 2.5, 3.0],
    //     [3.5, 4.0, 4.5],
    //     [5.0, 5.5, 6.0],
    //   ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_pool2DPixelAverage() {
    #if canImport(SwiftRTCuda)
      typealias Pixel = RGBA<UInt8>
      let image = array([
        [Pixel(0), Pixel(1), Pixel(2)],
        [Pixel(3), Pixel(4), Pixel(5)],
        [Pixel(6), Pixel(7), Pixel(8)],
      ])

      let avg = pool(x: image, windowSize: 3, padding: 1, op: .average)
      print(avg)
      XCTAssert(avg.shape == image.shape)
    // XCTAssert(
    //   same == [
    //     [2.0, 2.5, 3.0],
    //     [3.5, 4.0, 4.5],
    //     [5.0, 5.5, 6.0],
    //   ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_pool3DPixelAverage() {
    #if canImport(SwiftRTCuda)
      typealias Pixel = RGBA<UInt8>
      let volume = array([
        [
          [Pixel(0), Pixel(1), Pixel(2)],
          [Pixel(3), Pixel(4), Pixel(5)],
          [Pixel(6), Pixel(7), Pixel(8)],
        ],
        [
          [Pixel(9), Pixel(10), Pixel(11)],
          [Pixel(12), Pixel(13), Pixel(14)],
          [Pixel(15), Pixel(16), Pixel(17)],
        ],
      ])

      let avg = pool(x: volume, windowSize: 3, padding: 1, op: .average)
      print(avg)
      XCTAssert(avg.shape == volume.shape)
    // XCTAssert(
    //   same == [
    //     [2.0, 2.5, 3.0],
    //     [3.5, 4.0, 4.5],
    //     [5.0, 5.5, 6.0],
    //   ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolBatch2DPixelAverage() {
    #if canImport(SwiftRTCuda)
      typealias Pixel = RGBA<UInt8>
      let images = array([
        [
          [Pixel(0), Pixel(1), Pixel(2)],
          [Pixel(3), Pixel(4), Pixel(5)],
          [Pixel(6), Pixel(7), Pixel(8)],
        ],
        [
          [Pixel(9), Pixel(10), Pixel(11)],
          [Pixel(12), Pixel(13), Pixel(14)],
          [Pixel(15), Pixel(16), Pixel(17)],
        ],
      ])

      let avg = pool(batch: images, windowSize: 3, padding: 1, op: .average)
      print(avg)
      XCTAssert(avg.shape == images.shape)
    // XCTAssert(
    //   same == [
    //     [2.0, 2.5, 3.0],
    //     [3.5, 4.0, 4.5],
    //     [5.0, 5.5, 6.0],
    //   ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolBatch3DPixelAverage() {
    #if canImport(SwiftRTCuda)
      typealias Pixel = RGBA<UInt8>
      let volumes = array([
        // volume 0
        [
          [
            [Pixel(0), Pixel(1), Pixel(2)],
            [Pixel(3), Pixel(4), Pixel(5)],
            [Pixel(6), Pixel(7), Pixel(8)],
          ],
          [
            [Pixel(9), Pixel(10), Pixel(11)],
            [Pixel(12), Pixel(13), Pixel(14)],
            [Pixel(15), Pixel(16), Pixel(17)],
          ],
        ],
        // volume 1
        [
          [
            [Pixel(18), Pixel(19), Pixel(20)],
            [Pixel(21), Pixel(22), Pixel(23)],
            [Pixel(24), Pixel(25), Pixel(26)],
          ],
          [
            [Pixel(27), Pixel(28), Pixel(29)],
            [Pixel(30), Pixel(31), Pixel(32)],
            [Pixel(33), Pixel(34), Pixel(35)],
          ],
        ],
      ])

      let avg = pool(batch: volumes, windowSize: 3, padding: 1, op: .average)
      print(avg)
      XCTAssert(avg.shape == volumes.shape)
    // XCTAssert(
    //   same == [
    //     [2.0, 2.5, 3.0],
    //     [3.5, 4.0, 4.5],
    //     [5.0, 5.5, 6.0],
    //   ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolBatchAverage2D() {
    #if canImport(SwiftRTCuda)
      let a = array(0..<18, shape: (2, 3, 3))
      let avg = pool(batch: a, windowSize: 3, padding: 1, op: .average)
      XCTAssert(avg.shape == a.shape)
      XCTAssert(
        avg == [
          [
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
          ],
          [
            [11.0, 11.5, 12.0],
            [12.5, 13.0, 13.5],
            [14.0, 14.5, 15.0],
          ],
        ]
      )
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolAverage1D() {
    #if canImport(SwiftRTCuda)
      let a = array(0..<6)
      let avg = pool(x: a, windowSize: 3, padding: 1, op: .average)
      XCTAssert(avg == [0.5, 1.0, 2.0, 3.0, 4.0, 4.5])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolBatchAverage1D() {
    #if canImport(SwiftRTCuda)
      let a = array(0..<12, shape: (2, 6))
      let avg = pool(batch: a, windowSize: 3, padding: 1, op: .average)
      XCTAssert(
        avg == [
          [0.5, 1.0, 2.0, 3.0, 4.0, 4.5],
          [6.5, 7.0, 8.0, 9.0, 10.0, 10.5],
        ])

    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolAverage3D() {
    #if canImport(SwiftRTCuda)
      let a = array(0..<27, shape: (3, 3, 3))
      let avgrow = pool(x: a, windowSize: (1, 1, 3), padding: (0, 0, 1), op: .average)
      XCTAssert(
        avgrow == [
          [
            [0.5, 1.0, 1.5],
            [3.5, 4.0, 4.5],
            [6.5, 7.0, 7.5],
          ],
          [
            [9.5, 10.0, 10.5],
            [12.5, 13.0, 13.5],
            [15.5, 16.0, 16.5],
          ],
          [
            [18.5, 19.0, 19.5],
            [21.5, 22.0, 22.5],
            [24.5, 25.0, 25.5],
          ],
        ])

      let avgcol = pool(x: a, windowSize: (1, 3, 1), padding: (0, 1, 0), op: .average)
      XCTAssert(
        avgcol == [
          [
            [1.5, 2.5, 3.5],
            [3.0, 4.0, 5.0],
            [4.5, 5.5, 6.5],
          ],
          [
            [10.5, 11.5, 12.5],
            [12.0, 13.0, 14.0],
            [13.5, 14.5, 15.5],
          ],
          [
            [19.5, 20.5, 21.5],
            [21.0, 22.0, 23.0],
            [22.5, 23.5, 24.5],
          ],
        ])

      let avgdepth = pool(x: a, windowSize: (1, 3, 3), padding: (0, 1, 1), op: .average)
      XCTAssert(
        avgdepth == [
          [
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
          ],
          [
            [11.0, 11.5, 12.0],
            [12.5, 13.0, 13.5],
            [14.0, 14.5, 15.0],
          ],
          [
            [20.0, 20.5, 21.0],
            [21.5, 22.0, 22.5],
            [23.0, 23.5, 24.0],
          ],
        ])

      let avgvolume = pool(x: a, windowSize: 3, padding: 1, op: .average)
      XCTAssert(
        avgvolume == [
          [
            [6.5, 7.0, 7.5],
            [8.0, 8.5, 9.0],
            [9.5, 10.0, 10.5],
          ],
          [
            [11.0, 11.5, 12.0],
            [12.5, 13.0, 13.5],
            [14.0, 14.5, 15.0],
          ],
          [
            [15.5, 16.0, 16.5],
            [17.0, 17.5, 18.0],
            [18.5, 19.0, 19.5],
          ],
        ])
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolBatchAverage3D() {
    #if canImport(SwiftRTCuda)
      let a = array(0..<54, shape: (2, 3, 3, 3))
      let avg = pool(batch: a, windowSize: 3, padding: 1, op: .average)
      XCTAssert(avg.shape == a.shape)
      print(avg)
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolAverage2D() {
    #if canImport(SwiftRTCuda)
      // average rows
      do {
        let a = array(0..<9, shape: (3, 3))
        let avg = pool(x: a, windowSize: (1, 3), padding: (0, 1), op: .average)
        XCTAssert(avg.shape == a.shape)
        XCTAssert(
          avg == [
            [0.5, 1.0, 1.5],
            [3.5, 4.0, 4.5],
            [6.5, 7.0, 7.5],
          ])
      }

      // average cols
      do {
        let a = array(0..<9, shape: (3, 3))
        let avg = pool(x: a, windowSize: (3, 1), padding: (1, 0), op: .average)
        XCTAssert(avg.shape == a.shape)
        XCTAssert(
          avg == [
            [1.5, 2.5, 3.5],
            [3.0, 4.0, 5.0],
            [4.5, 5.5, 6.5],
          ])
      }

      do {
        let a = array([
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
        ])

        // same
        let same = pool(x: a, windowSize: 3, padding: 1, op: .average)
        XCTAssert(a.shape == same.shape)
        XCTAssert(
          same == [
            [2.0, 2.5, 3.0],
            [3.5, 4.0, 4.5],
            [5.0, 5.5, 6.0],
          ])

        // default is strides 1 padding 0
        let valid = pool(x: a, windowSize: 3, op: .average)
        XCTAssert(valid == [[4.0]])

        // using a configuration
        let config = PoolingConfig(x: a, windowSize: Shape2(3, 3), op: .average)
        var out = Tensor2(shape: config.shape, order: a.order)
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[4.0]])
      }

      do {
        let a = array(0..<25, shape: (5, 5))

        // same
        let same = pool(x: a, windowSize: 5, padding: 2, op: .average)
        XCTAssert(a.shape == same.shape)
        let expsame = array([
          [6.0, 6.5, 7.0, 7.5, 8.0],
          [8.5, 9.0, 9.5, 10.0, 10.5],
          [11.0, 11.5, 12.0, 12.5, 13.0],
          [13.5, 14.0, 14.5, 15.0, 15.5],
          [16.0, 16.5, 17.0, 17.5, 18.0],
        ])
        XCTAssert(almostEqual(same, expsame, tolerance: 0.001))

        // valid
        let valid = pool(x: a, windowSize: 5, op: .average)
        XCTAssert(valid == [[12.0]])

        // using a configuration
        let config = PoolingConfig(x: a, windowSize: 5, op: .average)
        var out = Tensor2(shape: config.shape, order: a.order)
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[12.0]])
      }
    #endif
  }

  //--------------------------------------------------------------------------
  func test_poolAveragePadding() {
    #if canImport(SwiftRTCuda)
      // 3D
      do {
        let a = ones(shape: (3, 3, 3))
        let avg = pool(x: a, windowSize: (1, 3, 1), padding: (1, 1, 1), op: .averagePadding)
        print(avg)
        XCTAssert(avg.shape == a.shape)
        let expavg = array([
          [
            [0.2962963, 0.44444445, 0.2962963],
            [0.44444445, 0.6666667, 0.44444445],
            [0.2962963, 0.44444445, 0.2962963],
          ],
          [
            [0.2962963, 0.44444445, 0.2962963],
            [0.44444445, 0.6666667, 0.44444445],
            [0.2962963, 0.44444445, 0.2962963],
          ],
        ])
        XCTAssert(almostEqual(avg, expavg, tolerance: 0.001))
      }

      // 2D
      do {
        let a = array([
          [0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
        ])

        // same
        let same = pool(x: a, windowSize: 3, padding: 1, op: .averagePadding)
        XCTAssert(a.shape == same.shape)
        XCTAssert(
          almostEqual(
            same,
            array([
              [0.8888889, 1.6666666, 1.3333334],
              [2.3333333, 4.0, 3.0],
              [2.2222223, 3.6666667, 2.6666667],
            ]),
            tolerance: 0.001))

        // valid
        let valid = pool(x: a, windowSize: 3, op: .averagePadding)
        XCTAssert(valid == [[4.0]])

        // using a configuration
        let config = PoolingConfig(x: a, windowSize: 3, op: .averagePadding)
        var out = Tensor2(shape: config.shape, order: a.order)
        currentQueue.pool(config, a, &out)
        XCTAssert(out == [[4.0]])
      }

      do {
        let a = array(0..<25, shape: (5, 5))

        // same
        let same = pool(x: a, windowSize: 5, padding: 1, op: .averagePadding)
        XCTAssert(a.shape == same.shape)
        let expsame = array([
          [2.1599998, 3.12, 4.2, 3.6, 2.8799999],
          [4.08, 5.7599998, 7.6, 6.3999996, 5.04],
          [6.6, 9.2, 12.0, 10.0, 7.7999997],
          [6.48, 8.96, 11.599999, 9.599999, 7.44],
          [5.7599998, 7.9199996, 10.2, 8.4, 6.48],
        ])
        XCTAssert(almostEqual(same, expsame, tolerance: 0.001))

        // valid
        let valid = pool(x: a, windowSize: 5, op: .averagePadding)
        XCTAssert(valid == [[12.0]])

        // using a configuration
        let config = PoolingConfig(x: a, windowSize: 5, op: .averagePadding)
        var out = Tensor2(shape: config.shape, order: a.order)
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
      let same = pool(x: a, windowSize: 3, padding: 1, op: .max)
      XCTAssert(a.shape == same.shape)
      XCTAssert(
        same == [
          [4.0, 5.0, 5.0],
          [7.0, 8.0, 8.0],
          [7.0, 8.0, 8.0],
        ])

      // valid
      let valid = pool(x: a, windowSize: 3, op: .max)
      XCTAssert(valid == [[8.0]])
    #endif
  }
}
