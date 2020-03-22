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
import XCTest
import Foundation
import SwiftRT

class test_npStyle: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_empty", test_empty),
    ]
    
    //==========================================================================
    // test_empty
    func test_empty() {
        // T0
        let _ = empty()
        let _ = empty(dtype: Int32.self)

        // T1
        let _ = empty(3)
        let _ = empty(3, order: .F)
        let _ = empty(3, dtype: Int32.self)
        let _ = empty(3, dtype: Int32.self, order: .F)

        // T2
        let _ = empty((2, 3))
        let _ = empty((2, 3), order: .F)
        let _ = empty((2, 3), dtype: Int32.self)
        let _ = empty((2, 3), dtype: Int32.self, order: .F)
    }
}
