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

class test_Fractals: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_juliaSet", test_juliaSet),
    ]

    // append and use a discrete async cpu device for these tests
    override func setUpWithError() throws {
        log.level = .diagnostic
    }

    override func tearDownWithError() throws {
        log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_juliaSet() {
        // parameters
        let iterations = 2048
        let size = (1024, 1025)
        let tolerance: Float = 4.0
        let C = Complex<Float>(-0.8, 0.156)
        let first = Complex<Float>(-1.7, -1.7)
        let last = Complex<Float>(1.7, 1.7)

//        let first = Complex<Float>(-1, -1)
//        let last = Complex<Float>(1, 1)

        print("size: \(size), iterations: \(iterations), " +
              "queue: \(currentQueue.name)")

        // test
        var Z = array(from: first, to: last, size)
        var divergence = full(size, iterations)

//        measure {
        let start = Date()

        pmap(&Z, &divergence) { Z, divergence in
            for i in 0..<1 {
                Z = multiply(Z, Z, add: C)
                divergence[abs(Z) .> tolerance] = min(divergence, i)
            }
        }

        print("elapsed: \(Date().timeIntervalSince(start))")
    }
}
