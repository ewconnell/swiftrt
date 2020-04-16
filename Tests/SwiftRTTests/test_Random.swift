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

class test_Random: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_randomUniform", test_randomUniform),
        ("test_randomNormal", test_randomNormal),
        ("test_randomTruncatedNormal", test_randomTruncatedNormal),
    ]
    
    //--------------------------------------------------------------------------
    // test_randomUniform
    func test_randomUniform() {
        let v = Tensor1<Float>(randomUniform: 10)
        XCTAssert(v.count > 0)
    }

    //--------------------------------------------------------------------------
    // test_randomNormal
    func test_randomNormal() {
        let v = Tensor1<Float>(randomNormal: 100)
        print(v.array)

        let someData = array(0..<100)
        let dataMean = mean(someData)
        let dataStd = dataMean //standardDeviation(someData)
        let weights = Tensor1<Float>(randomNormal: 100, mean: dataMean, standardDeviation: dataStd)
        print(weights)
    }

    //--------------------------------------------------------------------------
    // test_randomTruncatedNormal
    func test_randomTruncatedNormal() {
        let v = Tensor1<Float>(randomTruncatedNormal: 100)
        print(v.array)
    }
}
