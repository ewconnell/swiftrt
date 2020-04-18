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
        ("test_perfTensor1AddRange", test_perfTensor1AddRange),
        ("test_negativeIndexRelativeRange", test_negativeIndexRelativeRange),
        ("test_AssignDataToTensor3Item", test_AssignDataToTensor3Item),
        ("test_AssignDataToTensor3Range", test_AssignDataToTensor3Range),
        ("test_Tensor1Range", test_Tensor1Range),
        ("test_Tensor1RangeGradient", test_Tensor1RangeGradient),
        ("test_Tensor1WriteRange", test_Tensor1WriteRange),
        ("test_Tensor2Range", test_Tensor2Range),
    ]

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
    // test_writeToRepeated
    func test_writeToRepeated() {
        Context.log.level = .diagnostic
        // test through subscript
        var a = repeating(1, (2, 3))
        a[1, 1] = 42
        XCTAssert(a == [[1, 1, 1], [1, 42, 1]])

        // test assigning range through subscript
        a[0..<1, ...] = repeating(2, (1, 3))
        XCTAssert(a == [[2, 2, 2], [1, 42, 1]])
    }
    
    //==========================================================================
    // test_negativeIndexRelativeRange
    func test_negativeIndexRelativeRange() {
        let m = array(0..<9, (3, 3))
        let v = m[-2..+1, ...]
        XCTAssert(v == [[3, 4, 5]])
    }
    
    //==========================================================================
    // test_AssignDataToTensor3Item
    func test_AssignDataToTensor3Item() {
        var a3 = array(0..<24, (2, 3, 4))
        
        // assign a volume depth to item 0
        a3[0] = repeating(3, (1, 3, 4))
        XCTAssert(a3 == [
            [[ 3, 3, 3, 3], [ 3, 3, 3, 3], [ 3, 3, 3, 3]],
            [[12,13,14,15], [16,17,18,19], [20,21,22,23]],
        ])

        // assign via type expansion to item 1
        a3[1] = expand(dims: repeating(7, (3, 4)), axis: 0)
        XCTAssert(a3.array == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])

        let g = pullback(at: repeating(7, (3, 4)),
                         in: { expand(dims: $0, axis: 0) })(ones((1, 3, 4)))
        XCTAssert(g.flatArray == [Float](repeating: 1, count: 12))
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
        volume[1...1] = expand(dims: repeating(7, (3, 4)), axis: 0)
        XCTAssert(volume == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [7,7,7,7], [7,7,7,7]],
        ])
        
        // assign a row
        volume[1, 1, ...] = expand(dims: array(0..<4), axes: (0, 1))
        XCTAssert(volume == [
            [[3,3,3,3], [3,3,3,3], [3,3,3,3]],
            [[7,7,7,7], [0, 1, 2, 3], [7,7,7,7]],
        ])
    }
    
    //==========================================================================
    // test_Tensor1Range
    func test_Tensor1Range() {
        let a = array(0..<10)
        
        // from index 1 through the end
        XCTAssert(a[1...] == array(1...9))

        // through last element
        XCTAssert(a[...-1] == array(0...9))
        XCTAssert(a[...] == array(0...9))

        // up to the second to last element
        XCTAssert(a[..<-2] == array(0...7))

        // between 4 and 2 back from the end
        XCTAssert(a[-4..<-2] == array(6...7))

        // sliding window starting at 2 and extending 3 (i.e 2 + 3)
        XCTAssert(a[2..+3] == array(2...4))
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
        XCTAssert(m1[0..+2, ...] == [
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
        XCTAssert(v1.flatArray == [0, 1, 7, 7, 7, 5, 6])
        
        let v2 = array(1...6)
        let g = pullback(at: v2, in: { exp($0) })(ones(like: v2))
        XCTAssert(g == array([2.7182817, 7.389056, 20.085537,
                              54.59815, 148.41316, 403.4288]))
    }
}
