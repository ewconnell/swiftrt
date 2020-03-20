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
        ("test_vector", test_vector),
        ("test_matrix", test_matrix),
        ("test_RGBImage", test_RGBImage),
        ("test_RGBAImage", test_RGBAImage),
    ]
    
    //==========================================================================
    // test_vector
    // encodes and decodes
    func test_vector() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected: [Float] = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            let vector = Vector(expected)
            let jsonData = try jsonEncoder.encode(vector)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let vector2 = try decoder.decode(Vector.self, from: jsonData)
            XCTAssert(vector2 == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_matrix
    // encodes and decodes
    func test_matrix() {
        do {
            let jsonEncoder = JSONEncoder()
            let expected = (0..<10).map { Float($0) }
            let matrix = Matrix(2, 5, with: expected)
            let jsonData = try jsonEncoder.encode(matrix)
//            let jsonVectorString = String(data: jsonData, encoding: .utf8)!
//            print(jsonVectorString)
            let decoder = JSONDecoder()
            let matrix2 = try decoder.decode(Matrix.self, from: jsonData)
            XCTAssert(matrix2 == expected)
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
            typealias Image = SwiftRT.Matrix<Pixel>
            let jsonEncoder = JSONEncoder()
            let pixels = [Pixel(0, 0.5, 1), Pixel(0.25, 0.5, 0.75)]
            let image = Image(1, 2, with: pixels, name: "pixels")
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
            typealias Image = SwiftRT.Matrix<Pixel>
            let jsonEncoder = JSONEncoder()
            let pixels = [Pixel(0, 0.25, 0.5, 1), Pixel(0.25, 0.5, 0.75, 1)]
            let image = Image(1, 2, with: pixels, name: "pixels")
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
