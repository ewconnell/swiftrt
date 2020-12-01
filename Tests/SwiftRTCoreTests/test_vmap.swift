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

#if swift(>=5.3) && canImport(_Differentiation)
import _Differentiation
#endif

class test_vmap: XCTestCase {
  //==========================================================================
  static var allTests = [
    ("test_vmap3D", test_vmap3D),
  ]

  //--------------------------------------------------------------------------
  func test_vmap3D() {
    let a = array(0..<12, shape: (3, 4))
    print(a)
    let b = vmap(a, axis: 1) { col in
      let c = ones(shape: 3)
      let out = col + c
      print(out)
      return out
    }
    print(b)
  }
}
