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

class test_IterateView: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_ArrayProperty", test_ArrayProperty),
        ("test_Vector", test_Vector),
        ("test_Matrix", test_Matrix),
        ("test_Volume", test_Volume),
        ("test_VectorSubView", test_VectorSubView),
        ("test_MatrixSubView", test_MatrixSubView),
        ("test_VolumeSubView", test_VolumeSubView),
        ("test_perfVector", test_perfVector),
        ("test_perfMatrix", test_perfMatrix),
        ("test_perfVolume", test_perfVolume),
        ("test_repeatingValue", test_repeatingElement),
        ("test_repeatingRow", test_repeatingRow),
        ("test_repeatingCol", test_repeatingCol),
        ("test_repeatingColInVolume", test_repeatingColInVolume),
        ("test_repeatingMatrix", test_repeatingMatrix),
        ("test_repeatingMatrixSubView", test_repeatingMatrixSubView),
    ]
    
    //==========================================================================
    // test_ArrayProperty
    func test_ArrayProperty() {
        let vector = Vector(0..<6)
        XCTAssert(vector.array == [0, 1, 2, 3, 4, 5])

        let matrix = Matrix(2, 3, with: 0..<6)
        XCTAssert(matrix.array == [[0, 1, 2], [3, 4, 5]])

        let volume = Volume(2, 2, 3, with: 0..<12)
        XCTAssert(volume.array ==
            [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
    }
    
    //==========================================================================
    // test_Vector
    func test_Vector() {
        let count: Int32 = 10
        let expected = [Int32](0..<count)
        let vector = IndexVector(expected)
        XCTAssert(vector == expected)
    }
    
    //==========================================================================
    // test_Matrix
    func test_Matrix() {
        let expected = [Int32](0..<4)
        let matrix = IndexMatrix(2, 2, with: expected)
        XCTAssert(matrix == expected)
    }
    
    //==========================================================================
    // test_Volume
    func test_Volume() {
        let expected = [Int32](0..<24)
        let volume = IndexVolume(2, 3, 4, with: expected)
        XCTAssert(volume == expected)
    }

    //==========================================================================
    // test_VectorSubView
    func test_VectorSubView() {
        let vector = IndexVector(0..<10)
        let view = vector[2..<5]
        XCTAssert(view == [2, 3, 4])
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        let matrix = IndexMatrix(3, 4, with: 0..<12)
        let view = matrix[1..., 1..<3]
        XCTAssert(view == [
            5, 6,
            9, 10
        ])
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        let volume = IndexVolume(3, 3, 4, with: 0..<36)
        let view = volume[1..., 1..., 1...]

        XCTAssert(view == [
            17, 18, 19,
            21, 22, 23,

            29, 30, 31,
            33, 34, 35,
        ])
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        #if !DEBUG
        let count = 65535
        let vector = Vector(with: 0..<count)
        let elements = vector.bufferElements()
        var value: Float = 0
        
        self.measure {
            for _ in 0..<16 {
                value = 0
                for element in elements {
                    value += element
                }
            }
        }
        XCTAssert(value > 0)
        #endif
    }
    
    //==========================================================================
    // test_perfMatrix
    func test_perfMatrix() {
        #if !DEBUG
        let rows = 1024
        let cols = 64
        
        let matrix = Matrix(rows, cols, with: 0..<(rows * cols))
        let elements = matrix.bufferElements()
        var value: Float = 0
        
        self.measure {
            for _ in 0..<16 {
                value = 0
                for element in elements {
                    value += element
                }
            }
        }
        XCTAssert(value > 0)
        #endif
    }
    
    //==========================================================================
    // test_perfVolume
    func test_perfVolume() {
        #if !DEBUG
        let depths = 16
        let rows = 64
        let cols = 64
        
        let volume = IndexVolume(depths, rows, cols,
                                 with: 0..<(depths * rows * cols))
        let elements = volume.bufferElements()
        var value: Int32 = 0
        
        self.measure {
            for _ in 0..<16 {
                value = 0
                for element in elements {
                    value += element
                }
            }
        }
        XCTAssert(value > 0)
        #endif
    }
    
    //==========================================================================
    // test_repeatingElement
    func test_repeatingElement() {
        let matrix = IndexMatrix(element: 42).repeated(to: 2, 3)
        XCTAssert(matrix == [
            42, 42, 42,
            42, 42, 42,
        ])
    }
    
    //==========================================================================
    // test_repeatingRow
    func test_repeatingRow() {
        let matrix = IndexMatrix(1, 3, with: 0...2).repeated(to: 2, 3)
        XCTAssert(matrix == [
            0, 1, 2,
            0, 1, 2,
        ])
    }
    
    //==========================================================================
    // test_repeatingCol
    func test_repeatingCol() {
        let matrix = IndexMatrix(3, 1, with: 0...2).repeated(to: 3, 2)
        XCTAssert(matrix == [
            0, 0,
            1, 1,
            2, 2,
        ])
    }
    
    //==========================================================================
    // test_repeatingColInVolume
    func test_repeatingColInVolume() {
        let v = IndexVolume(1, 3, 1, with: [1, 0, 1]).repeated(to: 2, 3, 4)
        XCTAssert(v == [
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,

            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ])
    }
    
    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
        let v = IndexVolume([
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ]
        ]).repeated(to: 2, 3, 4)

        XCTAssert(v == [
            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,

            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,
        ])
    }
    
    //==========================================================================
    // test_repeatingMatrixSubView
    func test_repeatingMatrixSubView() {
        let matrix = IndexMatrix(3, 1, with: [1, 0, 1]).repeated(to: 3, 4)
        XCTAssert(matrix == [
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ])
        
        let view = matrix[1..<3, 1..<4]
        XCTAssert(view == [
            0, 0, 0,
            1, 1, 1,
        ])
    }
}
