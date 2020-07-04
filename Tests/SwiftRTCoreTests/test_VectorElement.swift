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

    //--------------------------------------------------------------------------
    // adds a value to each image pixel
    // TODO: RGB needs additional conformances to support + - * /
    func test_addPixel() {
        typealias Pixel = RGB<Float>
        typealias Image = TensorR2<Pixel>
        let value = Pixel(0.25, 0.5, 0.75)

        let pixels = [
            [Pixel(0, 0.5, 1), Pixel(0.25, 0.5, 0.75)],
            [Pixel(1, 1.5, 2), Pixel(1.25, 1.5, 1.75)]
        ]
        let image = array(pixels, name: "pixels")
        XCTAssert(image[1, 1].b == 1.75)
        
        let adjusted = image + value
        print(adjusted)
    }
}
