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
        ("test_juliaSetPmap", test_juliaSetPmap),
        ("test_juliaSetPmapFused", test_juliaSetPmapFused),
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
        
        
        var Z = array(from: first, to: last, size)
        var divergence = full(size, iterations)

        let start = Date()
        
        // 14.820s
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }

        print("time: \(Date().timeIntervalSince(start))")
    }

    //--------------------------------------------------------------------------
    func test_juliaSetPmap() {
        // parameters
        let iterations = 2048
        let size = (1024, 1025)
        let tolerance: Float = 4.0
        let C = Complex<Float>(-0.8, 0.156)
        let first = Complex<Float>(-1.7, -1.7)
        let last = Complex<Float>(1.7, 1.7)

        
        var Z = array(from: first, to: last, size)
        var divergence = full(size, iterations)

        // 0.959
        measure {
            pmap(&Z, &divergence, 16) { Z, divergence in
                for i in 0..<iterations {
                    Z = multiply(Z, Z, add: C)
                    divergence[abs(Z) .> tolerance] = min(divergence, i)
                }
            }
        }
    }

    func test_juliaSetPmapFused() {
        // parameters
        let iterations = 2048
        let size = (1024, 1025)
        let tolerance: Float = 4.0
        let C = Complex<Float>(-0.8, 0.156)
        let first = Complex<Float>(-1.7, -1.7)
        let last = Complex<Float>(1.7, 1.7)
        
        print("size: \(size), iterations: \(iterations), " +
                "queue: \(currentQueue.name)")
        
        var Z = array(from: first, to: last, size)
        var divergence = full(size, iterations)
        
        // 0.310
        measure {
            pmap(&Z, &divergence) { Z, divergence in
                julia(Z: Z, divergence: &divergence, C, tolerance, iterations)
            }
        }
    }
}

//==============================================================================
// user defined element wise function
@inlinable public func julia<E>(
    Z: TensorR2<Complex<E>>,
    divergence: inout TensorR2<E>,
    _ C: Complex<E>,
    _ tolerance: E,
    _ iterations: Int
) {
    let dname = divergence.name
    currentQueue.elementwise(&divergence, Z,
        "julia(Z: \(Z.name), divergence: \(dname), constant: \(C), " +
            "tolerance: \(tolerance), iterations: \(iterations))"
    ) { d, Z in
        var d = d, Z = Z
        for i in 0..<iterations {
            Z = Z * Z + C
            if abs(Z).real > tolerance { d = min(d, E.Value(exactly: i)!) }
        }
        return d
    }
}
