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

class test_StorageElement: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_Int1StorageSize", test_Int1StorageSize),
        ("test_Int1Add", test_Int1Add),
        ("test_Int1Bool", test_Int1Bool),
        ("test_Int1Reshape", test_Int1Reshape),
    ]
    
    override func setUpWithError() throws {
        // log.level = .diagnostic
    }

    override func tearDownWithError() throws {
        // log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_Int1StorageSize() {
        let a = array([
            [0, 0],
            [0, 1],
            [1, 0],
        ], type: UInt1.self)
        XCTAssert(a.read().count == 1)

        let b = array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], type: UInt1.self)
        XCTAssert(b.read().count == 1)

        let c = array([0, 1, 0, 1, 0, 1, 0], type: UInt1.self)
        XCTAssert(c.read().count == 1)
        
        let d = array([0, 1, 0, 1, 0, 1, 0, 1], type: UInt1.self)
        XCTAssert(d.read().count == 1)
        
        let e = array([0, 1, 0, 1, 0, 1, 0, 1, 0], type: UInt1.self)
        XCTAssert(e.read().count == 2)
    }
    
    //--------------------------------------------------------------------------
    func test_Int1Add() {
        let a = array([
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ], type: UInt1.self)
        
        let b = array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ], type: UInt1.self)

        let c = a + b
        XCTAssert(c == [
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
        ])
    }

    //--------------------------------------------------------------------------
    func test_Int1Bool() {
        let a = array([
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ], type: UInt1.self)
        
        let b = array(a, (4, 2), type: Bool.self)

        let c = array([
            [false, false],
            [false, true],
            [false, false],
            [true, true],
        ])
        XCTAssert(b == c)
    }

    //--------------------------------------------------------------------------
    func test_Int1Reshape() {
        let a = array([
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ], type: UInt1.self)
        
        let b = reshape(a, (2, 4))
        XCTAssert(b == [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ])
    }
}
