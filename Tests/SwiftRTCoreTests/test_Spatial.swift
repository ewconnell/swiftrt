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
    ("test_pool", test_pool),
  ]

  //--------------------------------------------------------------------------
  func test_pool() {
    let a = array(0..<6, (2, 3))
    let avg = pool(a, size: (3, 3), strides: (1, 1), pad: .same, op: .average)
    print(avg)
  }
}