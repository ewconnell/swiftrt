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
        ("test_SequentialViews", test_SequentialViews),
        ("test_ShapeCollection", test_ShapeCollection),
        ("test_transposed", test_transposed),
    ]

    //--------------------------------------------------------------------------
    // test_perfIndexShape1
    func test_perfIndexShape1() {
        #if !DEBUG
        let shape = Shape(Bounds1(1024 * 1024))
        var count = 0
        self.measure {
            for _ in 0..<10 {
                let array = shape.array
                count += array.last!
            }
        }
        XCTAssert(count > 0)
        #endif
    }

    func test_perfIndexShape2() {
        #if !DEBUG
        let shape = Shape(Bounds2(1024, 1024))
        var count = 0
        self.measure {
            for _ in 0..<10 {
                let array = shape.array
                count += array.last!
            }
        }
        XCTAssert(count > 0)
        #endif
    }

    func test_perfIndexShape3() {
        #if !DEBUG
        let shape = Shape(Bounds3(64, 128, 128))
        var count = 0
        self.measure {
            for _ in 0..<10 {
                let array = shape.array
                count += array.last!
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    func test_perfIndexShape4() {
        #if !DEBUG
        let shape = Shape(Bounds4(2, 32, 128, 128))
        var count = 0
        self.measure {
            for _ in 0..<10 {
                let array = shape.array
                count += array.last!
            }
        }
        XCTAssert(count > 0)
        #endif
    }
    
    func test_perfIndexShape5() {
        #if !DEBUG
        let shape = Shape(Bounds5(2, 2, 16, 128, 128))
        var count = 0
        self.measure {
            for _ in 0..<10 {
                let array = shape.array
                count += array.last!
            }
        }
        XCTAssert(count > 0)
        #endif
    }

    //--------------------------------------------------------------------------
    // test_SequentialViews
    let simdPerfIterations = 10000000
    
    func test_perfSIMD2() {
        #if !DEBUG
        let bounds = SIMD2(2, 3)
        let strides = SIMD2(3, 1)
        let pos = SIMD2(1, 2)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        XCTAssert(total > 0)
        #endif
    }

    func test_perfSIMD3() {
        #if !DEBUG
        let bounds = SIMD3(2, 3, 4)
        let strides = SIMD3(12, 3, 1)
        let pos = SIMD3(1, 2, 3)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
        #endif
    }
    
    func test_perfSIMD4() {
        #if !DEBUG
        let bounds = SIMD4(1, 2, 3, 4)
        let strides = SIMD4(24, 12, 3, 1)
        let pos = SIMD4(0, 1, 2, 3)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
        #endif
    }

    func test_perfSIMD5() {
        #if !DEBUG
        let bounds = SIMD5(1, 2, 3, 4, 5)
        let strides = SIMD5(120, 60, 20, 5, 1)
        let pos = SIMD5(0, 1, 2, 3, 4)
        var total = 0
        measure {
            for _ in 0..<simdPerfIterations {
                let span = ((bounds &- 1) &* strides).wrappedSum() + 1
                let count = bounds.indices.reduce(1) { $0 * bounds[$1] }
                let linear = (pos &* strides).wrappedSum()
                total += span + count + linear
            }
        }
        print(total)
        XCTAssert(total > 0)
        #endif
    }
    //--------------------------------------------------------------------------
    // test_SequentialViews
    func test_SequentialViews() {
        // vector views are always sequential
        let v = Vector(with: 0..<6)
        let subv = v[1...2]
        XCTAssert(subv.shape.isSequential)
        
        // a batch of rows are sequential
        let m = Matrix(4, 5)
        let mrows = m[1...2, ...]
        XCTAssert(mrows.shape.isSequential)

        // a batch of columns are not sequential
        let m1 = Matrix(4, 5)
        let mcols = m1[..., 1...2]
        XCTAssert(!mcols.shape.isSequential)
    }
    
    //--------------------------------------------------------------------------
    // test_ShapeCollection
    func test_ShapeCollection() {
        // repeating
        XCTAssert(Shape(Bounds1(3), strides: Bounds1(0)) == [0, 0, 0])
        XCTAssert(Shape(Bounds2(2, 3), strides: Bounds2(0, 1)) == [0, 1, 2, 0, 1, 2])
        XCTAssert(Shape(Bounds2(2, 3), strides: Bounds2(1, 0)) == [0, 0, 0, 1, 1, 1])

        // strided
        XCTAssert(Shape(Bounds1(5), strides: Bounds1(3)) == [0, 3, 6, 9, 12])
        XCTAssert(Shape(Bounds1(5), strides: Bounds1(3)) == [0, 3, 6, 9, 12])
        XCTAssert(Shape(Bounds2(2, 3), strides: Bounds2(6, 2)) == [0, 2, 4, 6, 8, 10])

        // dense
        XCTAssert(Shape(Bounds2(2, 3)) == [0, 1, 2, 3, 4, 5])
        XCTAssert(Shape(Bounds3(2, 3, 4)) == [Int](0..<24))
    }

    //--------------------------------------------------------------------------
    // test_transposed
    func test_transposed() {
        let volume = Volume(2,3,4).filledWithIndex()
        let transVolume = volume.transposed(with: Bounds3(2,1,0))
        XCTAssert(transVolume.array == [[[ 0.0, 12.0],
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
    // test_perfInitShape2
    func test_perfInitShape2() {
        #if !DEBUG
        var shape = Shape(Bounds2(1, 1))
        let index = Shape.Index(Bounds2.one, 5)
        var i = 0
        self.measure {
            for _ in 0..<1000000 {
                let a = Shape(bounds: Bounds2(3, 4))
                let b = a.columnMajor
                let ds = a == b ? b.dense : a.dense
                let positive = Shape.makePositive(bounds: Bounds2(1, -1))
                let c = Shape(bounds: positive)
                let r = Shape(bounds: Bounds2.one).repeated(to: a.bounds)
                let j = a.joined(with: [ds, c, r], alongAxis: 1)
                let t = j.transposed()
                shape = t
                i = shape[index]
            }
        }
        XCTAssert(shape.bounds == Bounds2(13, 3) && i > 0)
        #endif
    }
}
