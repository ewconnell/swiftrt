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

class test_VectorElement: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_addPixel", test_addPixel),
    ]

    override func setUpWithError() throws {
        // Context.log.level = .diagnostic
    }

    override func tearDownWithError() throws {
        // Context.log.level = .error
    }

    //--------------------------------------------------------------------------
    // adds a value to each image pixel
    // TODO: VectorElement types need additional conformance support
    // for common set operations like + - * /. For now the only support
    // is AdditiveArithmetic for FloatingPoint scalar types just to prove
    // the concept.
    func test_addPixel() {
        typealias Pixel = RGBA<Float>
        typealias Image = TensorR2<Pixel>
        
        // if not specified, alpha == 1 or Scalar.max for integer types
        let image = array([
            [Pixel(0, 0.5, 1), Pixel(0.25, 0.5, 0.75)],
            [Pixel(1, 1.5, 2), Pixel(1.25, 1.5, 1.75)]
        ], name: "pixels")
        XCTAssert(image[1, 1].b == 1.75)

        // do SIMD add
        let a = image + Pixel(0.25, 0.5, 0.75, 0)
        
        XCTAssert(a == [
            [Pixel(0.25, 1.0, 1.75), Pixel(0.5, 1.0, 1.5)],
            [Pixel(1.25, 2.0, 2.75), Pixel(1.5, 2.0, 2.5)]
        ])
    }
}
