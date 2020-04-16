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

class test_Codable: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_Tensor1", test_Tensor1),
        ("test_Tensor2", test_Tensor2),
        ("test_RGBImage", test_RGBImage),
        ("test_RGBAImage", test_RGBAImage),
    ]
    
    //==========================================================================
    // test_Tensor1
    // encodes and decodes
    func test_Tensor1() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected: [Float] = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            let a = array(expected)
            let jsonData = try jsonEncoder.encode(a)
            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
            print(jsonVectorString)
            let decoder = JSONDecoder()
            let b = try decoder.decode(Tensor1<Float>.self, from: jsonData)
            XCTAssert(b == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_Tensor2
    // encodes and decodes
    func test_Tensor2() {
        do {
            let jsonEncoder = JSONEncoder()
            let a = array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
            let jsonData = try jsonEncoder.encode(a)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let b = try decoder.decode(Tensor2<Float>.self, from: jsonData)
            XCTAssert(b == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_RGBImage
    // encodes and decodes
    func test_RGBImage() {
        do {
            typealias Pixel = RGB<Float>
            typealias Image = Tensor2<Pixel>
            let jsonEncoder = JSONEncoder()
            let pixels = [[Pixel(0, 0.5, 1), Pixel(0.25, 0.5, 0.75)]]
            var image = array(pixels)
            image.name = "pixels"
            let jsonData = try jsonEncoder.encode(image)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let image2 = try decoder.decode(Image.self, from: jsonData)
            XCTAssert(image2 == pixels)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_RGBAImage
    // encodes and decodes
    func test_RGBAImage() {
        do {
            typealias Pixel = RGBA<Float>
            typealias Image = Tensor2<Pixel>
            let jsonEncoder = JSONEncoder()
            let pixels = [[Pixel(0, 0.25, 0.5, 1), Pixel(0.25, 0.5, 0.75, 1)]]
            var image = array(pixels)
            image.name = "pixels"
            let jsonData = try jsonEncoder.encode(image)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let image2 = try decoder.decode(Image.self, from: jsonData)
            XCTAssert(image2 == pixels)
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
