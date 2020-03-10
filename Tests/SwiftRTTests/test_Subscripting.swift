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

class test_Subscripting: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_perfVectorAdd", test_perfVectorAdd),
        ("test_perfVectorAddRange", test_perfVectorAddRange),
        ("test_negativeIndexRelativeRange", test_negativeIndexRelativeRange),
        ("test_WriteToRepeated", test_WriteToRepeated),
        ("test_AssignDataToVolumeItem", test_AssignDataToVolumeItem),
        ("test_AssignDataToVolumeRange", test_AssignDataToVolumeRange),
        ("test_VectorRange", test_VectorRange),
        ("test_StridedRangeInForLoop", test_StridedRangeInForLoop),
        ("test_VectorRangeGradient", test_VectorRangeGradient),
        ("test_VectorSteppedRange", test_VectorSteppedRange),
        ("test_VectorWriteRange", test_VectorWriteRange),
        ("test_MatrixRange", test_MatrixRange),
    ]

    //==========================================================================
    // test_perfVectorAdd
    func test_perfVectorAdd() {
        #if !DEBUG
        let v1 = Vector(with: 1...20)
        let v2 = Vector(with: 1...20)
        var count: Float = 0
        
        measure {
            for _ in 0..<1000 {
                let r = v1 + v2
                count += r.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //==========================================================================
    // test_perfVectorAddRange
    func test_perfVectorAddRange() {
        #if !DEBUG
        let v1 = Vector(with: 1...20)
        let v2 = Vector(with: 1...20)
        var count: Float = 0
        
        measure {
            for _ in 0..<1000 {
                let r = v1[1...] + v2[1...]
                count += r.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //==========================================================================
    // test_negativeIndexRelativeRange
    func test_negativeIndexRelativeRange() {
        let m = Matrix(3, 3, with: 0..<9)
        let v = m[-2..|1, ...]
        XCTAssert(v == [3, 4, 5])
        
        let m1 = Matrix(5, 3, with: 0..<15)
        let v1 = m1[-4..|3..2, ...]
        XCTAssert(v1.array == [[3, 4, 5], [9, 10, 11]])
    }
    
    //==========================================================================
    // test_WriteToRepeated
    func test_WriteToRepeated() {
        // create a repeated value which only uses a single Element of storage
        var repeated = Matrix(repeating: 1, to: 2, 3)
        
        // writing to the repeated tensor causes the repeated data
        // to be fully realized now using 6 storage Elements
        let m1 = Matrix(2, 1, with: [41, 42])
        repeated[..., 1] = m1
        XCTAssert(repeated.array == [[1, 41, 1], [1, 42, 1]])

        var m2 = Matrix(repeating: 2.0, to: 2,2)
        m2[1, 0] = 30.0
        XCTAssert(m2 == [2, 2, 30, 2])
        m2[0] = Matrix(1,2, with: [3.0, 4.0])
        XCTAssert(m2 == [3, 4, 30, 2])
        m2[1,0] = 25.0
        XCTAssert(m2 == [3, 4, 25, 2])
    }
    
    //==========================================================================
    // test_AssignDataToVolumeItem
    func test_AssignDataToVolumeItem() {
        var volume = Volume(2, 3, 4, with: 0..<24)
        
        // assign a volume depth to item 0
        volume[0] = Volume(repeating: 3, to: 1, 3, 4)
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
        ])

        // assign via type expansion to item 1
        volume[1] = Volume(expanding: Matrix(repeating: 7, to: 3, 4))
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])
        
        do {
            let ones = Volume(repeating: 1, to: 1, 3, 4)
            let g = pullback(at: Matrix(repeating: 7, to: 3, 4),
                             in: { Volume(expanding: $0) })(ones)
            XCTAssert(g == [Float](repeating: 1, count: 24))
        }
    }
    
    //==========================================================================
    // test_AssignDataToVolumeRange
    func test_AssignDataToVolumeRange() {
        var volume = Volume(2, 3, 4, with: 0..<24)
        
        // assign a volume depth to item 0
        volume[0..<1] = Volume(repeating: 3, to: 1, 3, 4)
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
        ])
        
        // assign via type expansion to item 1
        volume[1...1] = Volume(expanding: Matrix(repeating: 7, to: 3, 4))
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])
        
        // assign a row
        volume[1, 1, ...] = Volume(expanding: Vector(0..<4), alongAxes: 0, 1)
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [0, 1, 2, 3], [7,7,7,7]],
        ])
    }
    
    //==========================================================================
    // test_VectorRange
    func test_VectorRange() {
        let vector = IndexVector(0..<10)
        
        // from index 1 through the end
        XCTAssert(vector[1...] == 1...9)

        // through last element
        XCTAssert(vector[...-1] == 0...9)
        XCTAssert(vector[...] == 0...9)

        // up to the second to last element
        XCTAssert(vector[..<-2] == 0...7)

        // between 4 and 2 back from the end
        XCTAssert(vector[-4..<-2] == 6...7)

        // the whole range stepping by 2
        XCTAssert(vector[(...)..2] == 0..<10..2)
        XCTAssert(vector[.....2] == 0..<10..2)

        // sliding window starting at 2 and extending 3 (i.e 2 + 3)
        XCTAssert(vector[2..|3] == 2...4)

        // sliding window starting at 2 and extending 5, stepped
        XCTAssert(vector[2..|5..2] == [2, 4])
    }

    //==========================================================================
    // test_StridedRangeInForLoop
    func test_StridedRangeInForLoop() {
        XCTAssert([Int](0..<12..3) == [0, 3, 6, 9])
        XCTAssert((0..<8..2).count == 4)
        XCTAssert((0.0..<2.0..0.5).count == 4)
        XCTAssert([Double](0.0..<2.0..0.5) == [0.0, 0.5, 1.0, 1.5])
    }
    
    //==========================================================================
    // test_VectorRangeGradient
    func test_VectorRangeGradient() {
        let v = Vector(0..<10)

        // simple range selection
        let range = v[1..<3]
        let ones = Vector(repeating: 1, like: range)
        let g = pullback(at: range, in: { exp($0) })(ones)
        XCTAssert(g == [2.7182817, 7.389056])

        // test expression gradient
        let range2 = v[2...] - v[1..<-1]
        let ones2 = Vector(repeating: 1, like: range2)
        let g2 = pullback(at: range2, in: { exp($0) })(ones2)
        XCTAssert(g2 == [2.7182817, 2.7182817, 2.7182817, 2.7182817,
                         2.7182817, 2.7182817, 2.7182817, 2.7182817])
    }

    //==========================================================================
    // test_VectorSteppedRange
    func test_VectorSteppedRange() {
        let vector = IndexVector(0...9)
        XCTAssert(vector[1..<2..2] == [1])
        XCTAssert(vector[1..<4..2] == [1, 3])
        XCTAssert(vector[..<4..2] == [0, 2])
        XCTAssert(vector[1...4..2] == [1, 3])
        XCTAssert(vector[1..<5..3] == [1, 4])
        XCTAssert(vector[1..<6..3] == [1, 4])
        XCTAssert(vector[..<8..3] == [0, 3, 6])
        XCTAssert(vector[1..<8..3] == [1, 4, 7])
        XCTAssert(vector[(...)..3] == [0, 3, 6, 9])
        XCTAssert(vector[(1...)..3] == [1, 4, 7])
    }

    //==========================================================================
    // test_MatrixRange
    func test_MatrixRange() {
        let m1 = IndexMatrix(3, 4, with: [
            0, 1,  2,  3,
            4, 5,  6,  7,
            8, 9, 10, 11
        ])
        
        let v1 = m1[1..<-1, ...3]
        XCTAssert(v1 == 4...6)

        // use negative row value to work from end and select row 1
        XCTAssert(m1[-2..<2, 1..<4] == 5...7)
        
        // sliding window starting at 0 and extending 2
        XCTAssert(m1[0..|2, ...] == [
            0, 1,  2,  3,
            4, 5,  6,  7,
        ])
    }

    //--------------------------------------------------------------------------
    // test_VectorWriteRange
    func test_VectorWriteRange() {
//        Platform.log.level = .diagnostic
        var v1 = Vector(0...6)
        let sevens = Vector(repeatElement(7, count: 3))
        v1[2...4] = sevens
        XCTAssert(v1 == [0, 1, 7, 7, 7, 5, 6])
        
        let v2 = Vector(1...6)
        let ones = Vector(repeating: 1, like: v2)
        let g = pullback(at: v2, in: { exp($0) })(ones)
        XCTAssert(g == [2.7182817, 7.389056, 20.085537,
                        54.59815, 148.41316, 403.4288])
    }
}
