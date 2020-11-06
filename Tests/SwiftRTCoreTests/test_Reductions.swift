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

class test_Reductions: XCTestCase {
  //==========================================================================
  // support terminal test run
  static var allTests = [
    ("test_gather", test_gather),
    ("test_abssum", test_abssum),
    ("test_sumTensor1", test_sumTensor1),
    ("test_sumTensor2", test_sumTensor2),
    ("test_sumTensor3AlongAxes", test_sumTensor3AlongAxes),
    ("test_minTensor3AlongAxes", test_minTensor3AlongAxes),
    ("test_maxTensor3AlongAxes", test_maxTensor3AlongAxes),
    ("test_all", test_all),
    ("test_any", test_any),
    ("test_meanTensor2", test_meanTensor2),
    ("test_min", test_min),
    ("test_max", test_max),
  ]


  //--------------------------------------------------------------------------
  // test_gather
  // TODO: get this verified
  func test_gather() {
    let a = array([
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
    ])
    let ai = array([0, 2], type: DeviceIndex.self)
    let b = gather(from: a, indices: ai)
    XCTAssert(
      b == [
        [0, 1, 2],
        [6, 7, 8],
      ])

    let c = gather(from: a, indices: ai, axis: 1)
    XCTAssert(
      c == [
        [0, 2],
        [3, 5],
        [6, 8],
      ])

    let g0 = gradient(at: ones(like: a)) {
      gather(from: $0 * a, indices: ai).sum().element
    }
    XCTAssert(
      g0 == [
        [0, 1, 2],
        [0, 0, 0],
        [6, 7, 8],
      ])

    let g1 = gradient(at: ones(like: a)) {
      gather(from: $0 * a, indices: ai, axis: -1).sum().element
    }
    XCTAssert(
      g1 == [
        [0, 0, 2],
        [3, 0, 5],
        [6, 0, 8],
      ])
  }

  //--------------------------------------------------------------------------
  // test_sumTensor1
  func test_sumTensor1() {
    let m = array([1, 2, 3, 4])
    let result = m.sum()
    XCTAssert(result.shape == [1])
    XCTAssert(result.element == 10)
  }

  //--------------------------------------------------------------------------
  // test_sumTensor2
  func test_sumTensor2() {
    let m = array([
      [0, 1],
      [2, 3],
      [4, 5],
    ])

    // sum all
    do {
      let result = m.sum()
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15)
    }

    do {
      let result = m.sum(axes: 0, 1)
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15)
    }

    // sum cols
    do {
      let result = m.sum(axes: 1)
      XCTAssert(result == [[1], [5], [9]])
    }

    // sum rows
    do {
      let result = m.sum(axes: 0)
      XCTAssert(result == [[6, 9]])
    }
  }

  //--------------------------------------------------------------------------
  // test_sumTensor3AlongAxes
  func test_sumTensor3AlongAxes() {
    let v = array(
      [
        [
          [10, 2],
          [3, 4],
          [5, 6],
        ],

        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      ])

    // sum depths
    let s0 = v.sum(axes: 0)
    XCTAssert(
      s0 == [
        [
          [11, 4],
          [6, 8],
          [10, 12],
        ]
      ])

    // sum rows
    XCTAssert(
      v.sum(axes: 1) == [
        [
          [18, 12]
        ],
        [
          [9, 12]
        ],
      ])

    // sum columns
    XCTAssert(
      v.sum(axes: 2) == [
        [
          [12],
          [7],
          [11],
        ],
        [
          [3],
          [7],
          [11],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  // test_minTensor3AlongAxes
  func test_minTensor3AlongAxes() {
    let v = array([
      [
        [10, 2],
        [3, 4],
        [5, -6],
      ],
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    ])

    // depths
    XCTAssert(
      v.min(axes: 0) == [
        [
          [1, 2],
          [3, 4],
          [5, -6],
        ]
      ])

    // rows
    XCTAssert(
      v.min(axes: 1) == [
        [
          [3, -6]
        ],
        [
          [1, 2]
        ],
      ])

    // columns
    XCTAssert(
      v.min(axes: 2) == [
        [
          [2],
          [3],
          [-6],
        ],
        [
          [1],
          [3],
          [5],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  // test_maxTensor3AlongAxes
  func test_maxTensor3AlongAxes() {
    let v = array([
      [
        [10, 2],
        [3, 4],
        [5, -6],
      ],
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    ])

    // max depths
    let vm = v.max(axes: 0)
    XCTAssert(
      vm == [
        [
          [10, 2],
          [3, 4],
          [5, 6],
        ]
      ])

    // max rows
    XCTAssert(
      v.max(axes: 1) == [
        [
          [10, 4]
        ],
        [
          [5, 6]
        ],
      ])

    // max columns
    XCTAssert(
      v.max(axes: 2) == [
        [
          [10],
          [4],
          [5],
        ],
        [
          [2],
          [4],
          [6],
        ],
      ])
  }

  //--------------------------------------------------------------------------
  func test_abssum() {
    let m = array([
      [0, -1],
      [-2, 3],
      [4, -5],
    ])

    // sum all
    do {
      let result = m.abssum()
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15)
    }

    do {
      let result = m.abssum(axes: 0, 1)
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15)
    }

    // sum cols
    do {
      let result = m.abssum(axes: 1)
      XCTAssert(result == [[1], [5], [9]])
    }

    // sum rows
    do {
      let result = m.abssum(axes: 0)
      XCTAssert(result == [[6, 9]])
    }
  }

  //--------------------------------------------------------------------------
  func test_all() {
    let a = array([true, true, true])
    XCTAssert(a.all().element == true)

    let a1 = array([true, false, true])
    XCTAssert(a1.all().element == false)

    let a2 = array([false, false, false])
    XCTAssert(a2.all().element == false)
  }

  //--------------------------------------------------------------------------
  func test_any() {
    let a = array([true, true, true])
    XCTAssert(a.any().element == true)

    let a1 = array([false, false, true])
    XCTAssert(a1.any().element == true)

    let a2 = array([false, false, false])
    XCTAssert(a2.any().element == false)
  }

  //----------------------------------------------------------------------
  // test_meanTensor2
  func test_meanTensor2() {
    let m = array([
      [0, 1],
      [2, 3],
      [4, 5],
    ])

    // mean all
    do {
      let result = m.mean()
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15 / 6)
    }

    do {
      let result = m.mean(axes: 0, 1)
      XCTAssert(result.shape == [1, 1])
      XCTAssert(result.element == 15 / 6)
    }

    // mean cols
    do {
      let result = m.mean(axes: 1)
      XCTAssert(result == [[0.5], [2.5], [4.5]])
    }

    // mean rows
    do {
      let result = m.mean(axes: 0)
      XCTAssert(result == [[2, 3]])
    }
  }

  //--------------------------------------------------------------------------
  func test_min() {
    let m = array([
      [-1, 3, -6],
      [1, -3, 6],
    ])
    XCTAssert(m.min().element == -6)
    // XCTAssert(m.min(axes: 0) == [[-1, -3, -6]])
    // XCTAssert(m.min(axes: 1) == [[-6], [-3]])
  }

  //--------------------------------------------------------------------------
  func test_max() {
    let m = array([
      [-1, 3, -6],
      [1, -3, 6],
    ])
    XCTAssert(m.max().element == 6)
    // XCTAssert(m.max(axes: 0) == [[1, 3, 6]])
    // XCTAssert(m.max(axes: 1) == [[3], [6]])
  }

}
