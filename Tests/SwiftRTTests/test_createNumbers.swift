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

class test_createNumbers: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_empty", test_empty),
        ("test_emptyLike", test_emptyLike),
        ("test_ones", test_ones),
        ("test_onesLike", test_onesLike),
        ("test_zeros", test_zeros),
        ("test_zerosLike", test_zerosLike),
        ("test_full", test_full),
        ("test_fullLike", test_fullLike),
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

    //==========================================================================
    // test_emptyLike
    func test_emptyLike() {
        let proto = empty((2, 3))
        
        let _ = empty(like: proto)
        let _ = empty(like: proto, shape: (6))
        let _ = empty(like: proto, shape: (1, 2, 3))
        let _ = empty(like: proto, order: .F)
        let _ = empty(like: proto, order: .F, shape: (1, 2, 3))
        let _ = empty(like: proto, dtype: Int32.self)
        let _ = empty(like: proto, dtype: Int32.self, shape: (1, 2, 3))
        let _ = empty(like: proto, dtype: Int32.self, order: .F, shape: (1, 2, 3))
    }
    
    //==========================================================================
    // test_ones
    func test_ones() {
        // T0
        let _ = ones()
        let _ = ones(dtype: Int32.self)

        // T1
        let _ = ones(3)
        let _ = ones(3, order: .F)
        let _ = ones(3, dtype: Int32.self)
        let _ = ones(3, dtype: Int32.self, order: .F)

        // T2
        let _ = ones((2, 3))
        let _ = ones((2, 3), order: .F)
        let _ = ones((2, 3), dtype: Int32.self)
        let _ = ones((2, 3), dtype: Int32.self, order: .F)
    }

    //==========================================================================
    // test_onesLike
    func test_onesLike() {
        let proto = ones((2, 3))
        
        let _ = ones(like: proto)
        let _ = ones(like: proto, shape: (6))
        let _ = ones(like: proto, shape: (1, 2, 3))
        let _ = ones(like: proto, order: .F)
        let _ = ones(like: proto, order: .F, shape: (1, 2, 3))
        let _ = ones(like: proto, dtype: Int32.self)
        let _ = ones(like: proto, dtype: Int32.self, shape: (1, 2, 3))
        let _ = ones(like: proto, dtype: Int32.self, order: .F, shape: (1, 2, 3))
    }
    
    //==========================================================================
    // test_zeros
    func test_zeros() {
        // T0
        let _ = zeros()
        let _ = zeros(dtype: Int32.self)

        // T1
        let _ = zeros(3)
        let _ = zeros(3, order: .F)
        let _ = zeros(3, dtype: Int32.self)
        let _ = zeros(3, dtype: Int32.self, order: .F)

        // T2
        let _ = zeros((2, 3))
        let _ = zeros((2, 3), order: .F)
        let _ = zeros((2, 3), dtype: Int32.self)
        let _ = zeros((2, 3), dtype: Int32.self, order: .F)
    }

    //==========================================================================
    // test_zerosLike
    func test_zerosLike() {
        let proto = zeros((2, 3))
        
        let _ = zeros(like: proto)
        let _ = zeros(like: proto, shape: (6))
        let _ = zeros(like: proto, shape: (1, 2, 3))
        let _ = zeros(like: proto, order: .F)
        let _ = zeros(like: proto, order: .F, shape: (1, 2, 3))
        let _ = zeros(like: proto, dtype: Int32.self)
        let _ = zeros(like: proto, dtype: Int32.self, shape: (1, 2, 3))
        let _ = zeros(like: proto, dtype: Int32.self, order: .F, shape: (1, 2, 3))
    }
    
    //==========================================================================
    // test_full
    func test_full() {
        // T0
        let _ = full(42)
        let _ = full(42, dtype: Int32.self)

        // T1
        let _ = full(3)
        let _ = full(3, 42, order: .F)
        let _ = full(3, 42, dtype: Int32.self)
        let _ = full(3, 42, dtype: Int32.self, order: .F)

        // T2
        let _ = full((2, 3), 42)
        let _ = full((2, 3), 42, order: .F)
        let _ = full((2, 3), 42, dtype: Int32.self)
        let _ = full((2, 3), 42, dtype: Int32.self, order: .F)
    }

    //==========================================================================
    // test_fullLike
    func test_fullLike() {
        let proto = empty((2, 3))
        
        let _ = full(like: proto, 42)
        let _ = full(like: proto, 42, shape: (6))
        let _ = full(like: proto, 42, shape: (1, 2, 3))
        let _ = full(like: proto, 42, order: .F)
        let _ = full(like: proto, 42, order: .F, shape: (1, 2, 3))
        let _ = full(like: proto, 42, dtype: Int32.self)
        let _ = full(like: proto, 42, dtype: Int32.self, shape: (1, 2, 3))
        let _ = full(like: proto, 42, dtype: Int32.self, order: .F, shape: (1, 2, 3))
    }
}
