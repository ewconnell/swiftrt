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

class test_Initialize: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_index", test_index),
        ("test_copy", test_copy),
    ]
    
    //--------------------------------------------------------------------------
    // test_index
    // tests creating an index and dump values
    func test_index() {
//        let i = array(0..<24, (2, 3, 4), dtype: Int.self)
////        print(i.array)
//        print(i[1, 0, ...])
//        print(i[1, 0, ...].flatArray)
//        print(i[1, 0, ...].array)
    }

    //--------------------------------------------------------------------------
    // test_copy
    // tests copying from source to destination view
    func test_copy() {
    }
}
