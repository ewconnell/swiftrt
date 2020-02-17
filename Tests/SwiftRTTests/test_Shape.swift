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

class test_Shape: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_transposed", test_transposed),
        ("test_perfShape2", test_perfShape2),
    ]

    //--------------------------------------------------------------------------
    // test_ShapeCollection
    func test_ShapeCollection() {
        let s = Shape1(extents: (5), strides: (3))
        print([Int](s))
    }

    //--------------------------------------------------------------------------
    // test_transposed
    func test_transposed() {
        let volume = Volume(2,3,4).filledWithIndex().transposed(with: (2,1,0))
        XCTAssert(volume.array == [[[ 0.0, 12.0],
                                    [ 4.0, 16.0],
                                    [ 8.0, 20.0]],
                                   
                                   [[ 1.0, 13.0],
                                    [ 5.0, 17.0],
                                    [ 9.0, 21.0]],
                                   
                                   [[ 2.0, 14.0],
                                    [ 6.0, 18.0],
                                    [10.0, 22.0]],
                                   
                                   [[ 3.0, 15.0],
                                    [ 7.0, 19.0],
                                    [11.0, 23.0]]])
    }
    
    //--------------------------------------------------------------------------
    // test_perfShape2
    func test_perfShape2() {
        #if !DEBUG
        using(Platform.synchronousCpu) {
            var shape = Shape2(extents: Shape2.zeros)
            let index = ShapeArray((1, 1))
            var i = 0
            self.measure {
                for _ in 0..<100000 {
                    let a = Shape2(extents: (3, 4))
                    let b = a.columnMajor
                    let ds = a == b ? b.dense : a.dense
                    let c = Shape2(extents:
                        Shape2.makePositive(dims: Shape2.Array((1, -1))))
                    let r = Shape2(extents: Shape2.ones).repeated(to: a.extents)
                    let j = a.joined(with: [ds, c, r], alongAxis: 1)
                    let t = j.transposed()
                    shape = t
                    i = shape.linearIndex(of: index)
                }
            }
            XCTAssert(shape.extents == Shape2.Array((13, 3)) && i > 0)
        }
        #endif
    }
}
