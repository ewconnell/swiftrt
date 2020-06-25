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

let expected9_9 = array([
    [5.522258, 6.21933, 6.916403, 7.6134734, 8.310547, 9.007618, 9.70469],
    [11.795906, 12.492979, 13.19005, 13.887125, 14.584194, 15.281267, 15.978339],
    [18.069555, 18.766626, 19.4637, 20.16077, 20.857843, 21.554914, 22.25199],
    [24.343203, 25.04028, 25.737349, 26.43442, 27.13149, 27.828562, 28.525637],
    [30.616854, 31.313925, 32.010998, 32.70807, 33.405136, 34.102215, 34.799286],
    [36.890503, 37.587574, 38.284645, 38.98172, 39.67879, 40.37586, 41.072933],
    [43.164154, 43.861217, 44.558292, 45.255363, 45.952435, 46.649513, 47.34658]
])

class test_Convolution: XCTestCase {
    static var allTests = [
        ("test_Tensor2", test_Tensor2),
        ("test_Image", test_Image),
    ]

    //--------------------------------------------------------------------------
    func test_Tensor2() {
//        let a = array(0..<81, (9, 9))
//        let conv = Convolution2(filterShape: [3, 3])
//        let b = conv(a)
//        print(b)
//        XCTAssert(b == expected9_9)
        
//        let c = array(0..<81, (9, 9), type: UInt8.self)
//        let convi = Convolution<Shape2,UInt8,Float>(filterShape: [3, 3])
//        let d = convi(c)
//        print(d)
    }

    //--------------------------------------------------------------------------
    // TODO: added specialized conv for FixedVector conforming types
    func test_Image() {
//        typealias Pixel = RGB<UInt8>
//        var image = empty((9, 9), type: Pixel.self)
//        let conv = Convolution<Shape2,Pixel,Float>(filterShape: [3, 3])
//        let a = conv(image)
    }
}
