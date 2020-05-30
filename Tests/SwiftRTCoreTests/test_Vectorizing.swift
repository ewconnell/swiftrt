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

class test_Vectorizing: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_perfAplusBSequential", test_perfAplusBSequential),
        ("test_perfAplusB_NonSequential", test_perfAplusB_NonSequential),
        ("test_perfReduceAll", test_perfReduceAll),
        ("test_perfReduceAny", test_perfReduceAny),
        ("test_perfReduceSum", test_perfReduceSum),
        ("test_perfReduceMean", test_perfReduceMean),
        ("test_perfReduceMin", test_perfReduceMin),
        ("test_perfReduceMax", test_perfReduceMax),
        ("test_perfAlessOrEqualBAny", test_perfAlessOrEqualBAny),
    ]
    
    //--------------------------------------------------------------------------
    func test_perfAplusBSequential() {
        #if !DEBUG
        let a = ones((1024, 1024))
        let b = ones((1024, 1024))
        var count: DType = 0

        // 0.0250
        self.measure {
            for _ in 0..<10 {
                count += (a + b).first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //--------------------------------------------------------------------------
    func test_perfAplusB_NonSequential() {
        #if !DEBUG
        let size = 1024
        let a = array(1...(size * size), (size, size))
        let b = array(1...(size * size), (size, size), order: .F)
        var count: DType = 0
        
        // 0.180
        // TODO: walk through this to improve if possible
        self.measure {
            for _ in 0..<10 {
                let result = a + b
                count = result.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }

    //--------------------------------------------------------------------------
    func test_perfReduceAll() {
        #if !DEBUG
        let size = 1024
        let x = full((size, size), true, type: Bool.self)
        var value: Bool = false
        
        // 0.00357s
        self.measure {
            for _ in 0..<10 {
                value = x.all().element
            }
        }
        
        XCTAssert(value == true)
        print(value)
        #endif
    }

    //--------------------------------------------------------------------------
    func test_perfReduceAny() {
        #if !DEBUG
        let size = 1024
        let x = full((size, size), true, type: Bool.self)
        var value: Bool = false
        
        // 0.00354s
        self.measure {
            for _ in 0..<10 {
                value = x.any().element
            }
        }
        
        XCTAssert(value == true)
        print(value)
        #endif
    }
    
    //--------------------------------------------------------------------------
    func test_perfReduceSum() {
        #if !DEBUG
        let size = 1024
        let x = array(1...(size * size), (size, size))
        var value: DType = 0
        
        // 0.010s
        self.measure {
            for _ in 0..<10 {
                value += x.sum().element
            }
        }

        XCTAssert(value > 0)
        print(value)
        #endif
    }

    //--------------------------------------------------------------------------
    func test_perfReduceMean() {
        #if !DEBUG
        let size = 1024
        let x = array(1...(size * size), (size, size))
        var value: DType = 0
        
        // 0.010s
        self.measure {
            for _ in 0..<10 {
                value += x.mean().element
            }
        }
        
        XCTAssert(value > 0)
        print(value)
        #endif
    }
    
    //--------------------------------------------------------------------------
    func test_perfReduceMin() {
        #if !DEBUG
        let size = 1024 * 1024
        let a = Tensor1(randomNormal: size)
        var value: DType = 0
        
        // 0.010s
        self.measure {
            for _ in 0..<10 {
                value = a.min().element
            }
        }
        
        XCTAssert(value != -1)
        #endif
    }
    
    //--------------------------------------------------------------------------
    func test_perfReduceMax() {
        #if !DEBUG
        let size = 1024 * 1024
        let a = Tensor1(randomNormal: size)
        var value: DType = 0
        
        // 0.010s
        self.measure {
            for _ in 0..<10 {
                value = a.max().element
            }
        }
        XCTAssert(value != -1)
        #endif
    }
    
    //--------------------------------------------------------------------------
    func test_perfAlessOrEqualBAny() {
        #if !DEBUG
        let size = 1024
        let a = array(1...(size * size), (size, size))
        let b = array(0..<(size * size), (size, size))
        var value = true

        // .00312s
        self.measure {
            value = (a .<= b).any().element
        }
        
        XCTAssert(value == false)
        #endif
    }
}
