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
        ("test_perfTensor1Add", test_perfTensor1Add),
        ("test_perfTensor1AddRange", test_perfTensor1AddRange),
        ("test_negativeIndexRelativeRange", test_negativeIndexRelativeRange),
        ("test_AssignDataToTensor3Item", test_AssignDataToTensor3Item),
        ("test_AssignDataToTensor3Range", test_AssignDataToTensor3Range),
        ("test_Tensor1Range", test_Tensor1Range),
        ("test_StridedRangeInForLoop", test_StridedRangeCollection),
        ("test_Tensor1RangeGradient", test_Tensor1RangeGradient),
//        ("test_Tensor1SteppedRange", test_Tensor1SteppedRange),
        ("test_Tensor1WriteRange", test_Tensor1WriteRange),
        ("test_Tensor2Range", test_Tensor2Range),
    ]

    //==========================================================================
    // test_perfTensor1Add
    func test_perfTensor1Add() {
        #if !DEBUG
        let a = array(1...20)
        let b = array(1...20)
        var count: Float = 0
        
        measure {
            for _ in 0..<1000 {
                let r = a + b
                count += r.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //==========================================================================
    // test_perfTensor1AddRange
    func test_perfTensor1AddRange() {
        #if !DEBUG
        let a = array(1...20)
        let b = array(1...20)
        var count: Float = 0
        
        measure {
            for _ in 0..<1000 {
                let r = a[1...] + b[1...]
                count += r.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //==========================================================================
    // test_negativeIndexRelativeRange
    func test_negativeIndexRelativeRange() {
        let m = array(0..<9, (3, 3))
        let v = m[-2..|1, ...]
        XCTAssert(v == [[3, 4, 5]])

        // STEPPED RANGE TEST
//        let m1 = array(0..<15, (5, 3))
//        let v1 = m1[-4..|3..2, ...]
//        XCTAssert(v1 == [[3, 4, 5], [9, 10, 11]])
    }
    
    //==========================================================================
    // test_perfTensorSubview
    func test_perfTensorSubview() {
        #if !DEBUG
        let a = array(0..<6, (2, 3))
        var count: Float = 0
        
        measure {
            for _ in 0..<100000 {
                let view = a[1, ...]
                count += view.first
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    //==========================================================================
    // test_AssignDataToTensor3Item
    func test_AssignDataToTensor3Item() {
        var volume = array(0..<24, (2, 3, 4))
        
        // assign a volume depth to item 0
        volume[0] = repeating(3, (1, 3, 4))
        XCTAssert(volume == [
            [[ 3, 3, 3, 3], [ 3, 3, 3, 3], [ 3, 3, 3, 3]],
            [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
        ])

        // assign via type expansion to item 1
        volume[1] = Tensor3(expanding: repeating(7, (3, 4)))
        XCTAssert(volume.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])
        
        do {
            let g = pullback(at: repeating(7, (3, 4)),
                             in: { Tensor3(expanding: $0) })(ones((1, 3, 4)))
            XCTAssert(g.flat == [Float](repeating: 1, count: 12))
        }
    }
    
    //==========================================================================
    // test_AssignDataToTensor3Range
    func test_AssignDataToTensor3Range() {
        var volume = array(0..<24, (2, 3, 4))

        // assign a volume depth to item 0
        volume[0..<1] = repeating(3, (1, 3, 4))
        XCTAssert(volume == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
        ])
        
        // assign via type expansion to item 1
        volume[1...1] = Tensor3(expanding: repeating(7, (3, 4)))
        XCTAssert(volume == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])
        
        // assign a row
        volume[1, 1, ...] = Tensor3(expanding: array(0..<4), alongAxes: 0, 1)
        XCTAssert(volume == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [0, 1, 2, 3], [7,7,7,7]],
        ])
    }
    
    //==========================================================================
    // test_Tensor1Range
    func test_Tensor1Range() {
        let vector = array(0..<10)
        
        // from index 1 through the end
        XCTAssert(vector[1...] == array(1...9))

        // through last element
        XCTAssert(vector[...-1] == array(0...9))
        XCTAssert(vector[...] == array(0...9))

        // up to the second to last element
        XCTAssert(vector[..<-2] == array(0...7))

        // between 4 and 2 back from the end
        XCTAssert(vector[-4..<-2] == array(6...7))

        // sliding window starting at 2 and extending 3 (i.e 2 + 3)
        XCTAssert(vector[2..|3] == array(2...4))

        // STEPPED RANGE TEST
        // the whole range stepping by 2
//        XCTAssert(vector[(...)..2] == array(0..<10..2))
//        XCTAssert(vector[.....2] == array(0..<10..2))
//
//        // sliding window starting at 2 and extending 5, stepped
//        XCTAssert(vector[2..|5..2] == array([2, 4]))
    }

    //==========================================================================
    // test_StridedRangeCollection
    func test_StridedRangeCollection() {
        XCTAssert([Int](0..<12..3) == [0, 3, 6, 9])
        XCTAssert((0..<8..2).count == 4)
        XCTAssert((0.0..<2.0..0.5).count == 4)
        XCTAssert([Double](0.0..<2.0..0.5) == [0.0, 0.5, 1.0, 1.5])
    }
    
    //==========================================================================
    // test_Tensor1RangeGradient
    func test_Tensor1RangeGradient() {
        let v = array(0..<10)

        // simple range selection
        let range = v[1..<3]
        let g = pullback(at: range, in: { exp($0) })(ones(like: range))
        XCTAssert(g == array([2.7182817, 7.389056]))

        // test expression gradient
        let range2 = v[2...] - v[1..<-1]
        let g2 = pullback(at: range2, in: { exp($0) })(ones(like: range2))
        XCTAssert(g2 == array([2.7182817, 2.7182817, 2.7182817, 2.7182817,
                               2.7182817, 2.7182817, 2.7182817, 2.7182817]))
    }

    //==========================================================================
    // test_Tensor1SteppedRange
//    func test_Tensor1SteppedRange() {
//        let vector = array(0...9, dtype: Int.self)
//        XCTAssert(vector[1..<2..2] == [1])
//        XCTAssert(vector[1..<4..2] == [1, 3])
//        XCTAssert(vector[..<4..2] == [0, 2])
//        XCTAssert(vector[1...4..2] == [1, 3])
//        XCTAssert(vector[1..<5..3] == [1, 4])
//        XCTAssert(vector[1..<6..3] == [1, 4])
//        XCTAssert(vector[..<8..3] == [0, 3, 6])
//        XCTAssert(vector[1..<8..3] == [1, 4, 7])
//        XCTAssert(vector[(...)..3] == [0, 3, 6, 9])
//        XCTAssert(vector[(1...)..3] == [1, 4, 7])
//    }

    //==========================================================================
    // test_Tensor2Range
    func test_Tensor2Range() {
        let m1 = array([
            0, 1,  2,  3,
            4, 5,  6,  7,
            8, 9, 10, 11
        ], (3, 4))
        
        let v1 = m1[1..<-1, ..<3]
        XCTAssert(v1 == [[4, 5, 6]])

        // use negative row value to work from end and select row 1
        XCTAssert(m1[-2..<2, 1..<4] == [[5, 6, 7]])
        
        // sliding window starting at 0 and extending 2
        XCTAssert(m1[0..|2, ...] == [
            [0, 1,  2,  3],
            [4, 5,  6,  7],
        ])
    }

    //--------------------------------------------------------------------------
    // test_Tensor1WriteRange
    func test_Tensor1WriteRange() {
//        Platform.log.level = .diagnostic
        var v1 = array(0...6)
        let sevens = repeating(7, (3))
        v1[2...4] = sevens
        XCTAssert(v1.flat == [0, 1, 7, 7, 7, 5, 6])
        
        let v2 = array(1...6)
        let g = pullback(at: v2, in: { exp($0) })(ones(like: v2))
        XCTAssert(g == array([2.7182817, 7.389056, 20.085537,
                              54.59815, 148.41316, 403.4288]))
    }
}
