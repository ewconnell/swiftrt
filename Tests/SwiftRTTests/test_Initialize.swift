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

class test_Initialize: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_copy", test_copy),
        ("test_copyOnWrite", test_copyOnWrite),
        ("test_columnMajorDataView", test_columnMajorDataView),
        ("test_indenting", test_indenting),
        ("test_padding", test_padding),
        ("test_perfCreateTensorArray", test_perfCreateTensorArray),
        ("test_perfCreateMatrix", test_perfCreateMatrix),
        ("test_perfReadOnlyAccess", test_perfReadOnlyAccess),
        ("test_perfReadWriteAccess", test_perfReadWriteAccess),
        ("test_concatMatrixRows", test_concatMatrixRows),
        ("test_concatMatrixCols", test_concatMatrixCols),
        ("test_repeatElement", test_repeatElement),
        ("test_repeatRowVector", test_repeatRowVector),
        ("test_repeatColVector", test_repeatColVector),
    ]
    
    //--------------------------------------------------------------------------
    // test_copy
    // tests copying from source to destination view
    func test_copy() {
        let v1 = IndexVector(with: 1...3)
        var v2 = IndexVector(with: repeatElement(0, count: 3))
        SwiftRT.copy(from: v1, to: &v2)
        XCTAssert(v1 == [1, 2, 3])
    }
    
    //--------------------------------------------------------------------------
    // test_copyOnWrite
    // NOTE: uses the default queue
    func test_copyOnWrite() {
//        Platform.log.level = .diagnostic
        let m1 = Matrix(3, 2).filledWithIndex()
        XCTAssert(m1[1, 1] == 3)
        
        // copy view sharing the same tensor array
        var m2 = m1
        XCTAssert(m2[1, 1] == 3)
        
        // mutate m2
        m2[1, 1] = 7
        // m1's data should be unchanged
        XCTAssert(m1[1, 1] == 3)
        XCTAssert(m2[1, 1] == 7)
    }
    
    //--------------------------------------------------------------------------
    // test_columnMajorDataView
    // NOTE: uses the default queue
    //   0, 1,
    //   2, 3,
    //   4, 5
    func test_columnMajorDataView() {
        let cmMatrix = IndexMatrix(3, 2, with: [0, 2, 4, 1, 3, 5],
                                   layout: .columnMajor)
        XCTAssert(cmMatrix == 0..<6)
    }

    //--------------------------------------------------------------------------
    // test_indenting
    func test_indenting() {
        let v = Vector(with: 0..<4)
        let m = Matrix(indenting: v)
        XCTAssert(m.extents == [1, v.count])
    }
    
    //--------------------------------------------------------------------------
    // test_padding
    func test_padding() {
        let v = Vector(with: 0..<4)
        let m = Matrix(padding: v)
        XCTAssert(m.extents == [v.count, 1])
    }
    
    //--------------------------------------------------------------------------
    // test_perfCreateTensorArray
    func test_perfCreateTensorArray() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            let iterations = 10000
            var count = 0
            measure {
                for i in 1...iterations {
                    let array = TensorArray<Float>(count: i, name: "")
                    count = array.count
                }
            }
            XCTAssert(count == iterations)
        }
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_perfCreateMatrix
    func test_perfCreateMatrix() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            let iterations = 10000
            var count = 0
            measure {
                for i in 1...iterations {
                    let matrix = Matrix(1, i)
                    count = matrix.count
                }
            }
            XCTAssert(count == iterations)
        }
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_perfReadOnlyAccess
    func test_perfReadOnlyAccess() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            let iterations = 100000
            var value: Float = 0
            let matrix = Matrix(2, 2, with: 1...4)
            
            measure {
                do {
                    for _ in 1...iterations {
                        value = try matrix.readOnly()[0]
                    }
                } catch {
                    XCTFail()
                }
            }
            XCTAssert(value == 1)
        }
        #endif
    }
    
    //--------------------------------------------------------------------------
    // test_perfReadWriteAccess
    func test_perfReadWriteAccess() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            let iterations = 100000
            let value: Float = 1
            var matrix = Matrix(2, 2, with: 1...4)
            
            measure {
                do {
                    for _ in 1...iterations {
                        try matrix.readWrite()[0] = value
                    }
                    XCTAssert(try matrix.readWrite()[0] == value)
                } catch {
                    XCTFail()
                }
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // test_repeatElement
    func test_repeatElement() {
        let value: Int32 = 42
        let volume = IndexVolume(element: value).repeated(to: (2, 3, 10))
        let expected = [Int32](repeating: value, count: volume.count)
        XCTAssert(volume == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_repeatRowVector
    func test_repeatRowVector() {
        let matrix = IndexMatrix(1, 5, with: 0...4).repeated(to: (5, 5))
        XCTAssert(matrix == [
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
            0, 1, 2, 3, 4,
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_repeatColVector
    func test_repeatColVector() {
        let matrix = IndexMatrix(5, 1, with: 0...4).repeated(to: (5, 5))
        XCTAssert(matrix == [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3, 3, 3, 3, 3,
            4, 4, 4, 4, 4,
        ])
    }
    
    //--------------------------------------------------------------------------
    // test_concatMatrixRows
    func test_concatMatrixRows() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c3 = Matrix(concatenating: t1, t2)
        XCTAssert(c3.extents == [4, 3])
        XCTAssert(c3 == [
            1,  2,  3,
            4,  5,  6,
            7,  8,  9,
            10, 11, 12,
        ])
    }

    //--------------------------------------------------------------------------
    // test_concatMatrixCols
    func test_concatMatrixCols() {
        let t1 = Matrix(2, 3, with: 1...6)
        let t2 = Matrix(2, 3, with: 7...12)
        let c3 = Matrix(concatenating: t1, t2, alongAxis: 1)
        XCTAssert(c3.extents == [2, 6])
        XCTAssert(c3 == [
            1,  2,  3, 7,  8,  9,
            4,  5,  6, 10, 11, 12,
        ])
    }
}
