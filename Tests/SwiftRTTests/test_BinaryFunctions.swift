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
import Complex

//==============================================================================
// platform independent tests
fileprivate extension ComputePlatform {
    //--------------------------------------------------------------------------
    // test_addSubMulDivComplex
    func test_addSubMulDivComplex() {
        let data: [Complex<Float>] = [1, 2, 3, 4]
        let cm1 = ComplexMatrix(2, 2, elements: data)
        let cm2 = ComplexMatrix(2, 2, elements: data)
        let ones = ComplexMatrix(repeating: 1, like: cm1)
        
        // add a scalar
        XCTAssert(cm1 + 1 == [2, 3, 4, 5])
        
        // add tensors
        XCTAssert(cm1 + cm2 == [2, 4, 6, 8])
        
        // subtract a scalar
        XCTAssert(cm1 - 1 == [0, 1, 2, 3])
        
        // subtract tensors
        XCTAssert(cm1 - cm2 == [0, 0, 0, 0])
        
        // mul a scalar
        XCTAssert(cm1 * 2 == [2, 4, 6, 8])
        
        // mul tensors
        XCTAssert(cm1 * cm2 == [1, 4, 9, 16])
        
        // divide by a scalar
        let divExpected = [0.5, 1, 1.5, 2].map { Complex<Float>($0) }
        XCTAssert(cm1 / 2 == divExpected)
        
        // divide by a tensor
        XCTAssert(cm1 / cm2 == [1, 1, 1, 1])
        
        // test add derivative
        do {
            let (g1, g2) = pullback(at: cm1, cm2, in: { $0 + $1 })(ones)
            XCTAssert(g1 == [1, 1, 1, 1])
            XCTAssert(g2 == [1, 1, 1, 1])
        }
        
        do {
            let (g1, g2) = pullback(at: cm1, cm2, in: { $0 - $1 })(ones)
            XCTAssert(g1 == [1, 1, 1, 1])
            XCTAssert(g2 == [-1, -1, -1, -1])
        }
        do {
            let (g1, g2) = pullback(at: cm1, cm2, in: { $0 * $1 })(ones)
            XCTAssert(g1 == [1, 2, 3, 4])
            XCTAssert(g2 == [1, 2, 3, 4])
        }
        do {
            let (g1, g2) = pullback(at: cm1, cm2, in: { $0 / $1 })(ones)
            let data = [1, 0.5, 0.333333343, 0.25].map { Complex<Float>($0) }
            let g1Expected = ComplexMatrix(2, 2, elements: data)
            let g1sumdiff = sum(g1 - g1Expected).element
            XCTAssert(Swift.abs(g1sumdiff.real) <= 1e-6 && g1sumdiff.imaginary == 0)
            
            let g2Expected = -ComplexMatrix(2, 2, elements: data)
            let g2sumdiff = sum(g2 - g2Expected).element
            XCTAssert(Swift.abs(g2sumdiff.real) <= 1e-6 && g2sumdiff.imaginary == 0)
        }
    }
    
    //--------------------------------------------------------------------------
    // test_add
    func test_add() {
        let m1 = Matrix(3, 2, with: 0..<6)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 + m2
        XCTAssert(result == [0, 2, 4, 6, 8, 10])
        
        let (g1, g2) = gradient(at: m1, m2, in: { $0 + $1 })
        XCTAssert(g1 == [1, 1, 1, 1, 1, 1])
        XCTAssert(g2 == [1, 1, 1, 1, 1, 1])
    }
    
    //--------------------------------------------------------------------------
    // test_addInt32
    func test_addInt32() {
        let m1 = IndexMatrix(3, 2, with: 0..<6)
        let m2 = IndexMatrix(3, 2, with: 0..<6)
        let result = m1 + m2
        XCTAssert(result == [0, 2, 4, 6, 8, 10])
    }
    
    //--------------------------------------------------------------------------
    // test_addUInt8
    func test_addUInt8() {
        let m1 = MatrixType<UInt8>(3, 2, with: 0..<6)
        let m2 = MatrixType<UInt8>(3, 2, with: 0..<6)
        let result = m1 + m2
        XCTAssert(result == [0, 2, 4, 6, 8, 10])
    }
    
    //--------------------------------------------------------------------------
    // test_addScalar
    func test_addScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 + 1
        let expected: [Float] = [2, 3, 4, 5, 6, 7]
        XCTAssert(result == expected)
        
        let result2 = 1 + m1
        XCTAssert(result2 == expected)
    }
    
    //--------------------------------------------------------------------------
    // test_addAndAssign
    func test_addAndAssign() {
        var m1 = Matrix(3, 2, with: 0...5)
        m1 += 2
        XCTAssert(m1 == [2, 3, 4, 5, 6, 7])
    }
    
    //--------------------------------------------------------------------------
    // test_subtract
    func test_subtract() {
        let m1 = Matrix(3, 2, with: 1..<7)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 - m2
        XCTAssert(result == [1, 1, 1, 1, 1, 1])
        
        let (g1, g2) = gradient(at: m1, m2, in: { $0 - $1 })
        XCTAssert(g1 == [1, 1, 1, 1, 1, 1])
        XCTAssert(g2 == [-1, -1, -1, -1, -1, -1])
    }
    
    //--------------------------------------------------------------------------
    // test_subtractScalar
    func test_subtractScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 - 1
        XCTAssert(result == [0, 1, 2, 3, 4, 5])
        
        let result2 = 1 - m1
        XCTAssert(result2 == [0, -1, -2, -3, -4, -5])
    }
    
    //--------------------------------------------------------------------------
    // test_subtractVector
    func test_subtractVector() {
        let m1 = Matrix(3, 2, with: [
            1, 2,
            3, 4,
            5, 6
        ])
        let col = Matrix(3, 1, with: 0...2).repeated(to: (3, 2))
        
        let result = m1 - col
        let expected: [Float] = [
            1, 2,
            2, 3,
            3, 4
        ]
        XCTAssert(result == expected)
        
        let result2 = col - m1
        let expected2: [Float] = [
            -1, -2,
            -2, -3,
            -3, -4
        ]
        XCTAssert(result2 == expected2)
    }
    
    //--------------------------------------------------------------------------
    // test_subtractAndAssign
    func test_subtractAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 -= 1
        XCTAssert(m1 == [0, 1, 2, 3, 4, 5])
    }
    
    //--------------------------------------------------------------------------
    // test_mul
    func test_mul() {
        let m1 = Matrix(3, 2, with: 0..<6)
        let m2 = Matrix(3, 2, with: 0..<6)
        let result = m1 * m2
        XCTAssert(result == [0, 1, 4, 9, 16, 25])
        
        let (g1, g2) = gradient(at: m1, m2, in: { $0 * $1 })
        XCTAssert(g1 == [0, 1, 2, 3, 4, 5])
        XCTAssert(g2 == [0, 1, 2, 3, 4, 5])
    }
    
    //--------------------------------------------------------------------------
    // test_mulScalar
    func test_mulScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 * 2
        XCTAssert(result == [2, 4, 6, 8, 10, 12])
    }
    
    //--------------------------------------------------------------------------
    // test_mulAndAssign
    func test_mulAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 *= 2
        XCTAssert(m1 == [2, 4, 6, 8, 10, 12])
    }
    
    //--------------------------------------------------------------------------
    // test_div
    func test_div() {
        let m1 = Matrix(3, 2, with: [1, 4, 9, 16, 25, 36])
        let m2 = Matrix(3, 2, with: 1...6)
        let result = m1 / m2
        XCTAssert(result == [1, 2, 3, 4, 5, 6])
        
        do {
            let (g1, g2) = gradient(at: m1, m2, in: { $0 / $1 })
            let g1Expected = Matrix(3, 2, with:
                [1, 0.5, 0.3333333, 0.25, 0.2, 0.1666666])
            XCTAssert(abssum(g1 - g1Expected).element <= 1e-6)
            XCTAssert(g2 == [-1, -1, -1, -1, -1, -1])
        }
    }
    
    //--------------------------------------------------------------------------
    // test_divScalar
    func test_divScalar() {
        let m1 = Matrix(3, 2, with: 1...6)
        let result = m1 / 2
        XCTAssert(result == [0.5, 1, 1.5, 2, 2.5, 3])
    }
    
    //--------------------------------------------------------------------------
    // test_divAndAssign
    func test_divAndAssign() {
        var m1 = Matrix(3, 2, with: 1...6)
        m1 /= 2
        XCTAssert(m1 == [0.5, 1, 1.5, 2, 2.5, 3])
    }

}

//==============================================================================
// Cpu test runs
final class test_BinaryFunctionsCpu: XCTestCase {
    override class func setUp() {
        // set platform type to test
        Current.platform = Platform<CpuService>()
    }

    static var allTests = [
        ("test_addSubMulDivComplex", test_addSubMulDivComplex),
        ("test_add", test_add),
        ("test_addInt32", test_addInt32),
        ("test_addUInt8", test_addUInt8),
        ("test_addScalar", test_addScalar),
        ("test_addAndAssign", test_addAndAssign),

        ("test_subtract", test_subtract),
        ("test_subtractScalar", test_subtractScalar),
        ("test_subtractVector", test_subtractVector),
        ("test_subtractAndAssign", test_subtractAndAssign),

        ("test_mul", test_mul),
        ("test_mulScalar", test_mulScalar),
        ("test_mulAndAssign", test_mulAndAssign),

        ("test_div", test_div),
        ("test_divScalar", test_divScalar),
        ("test_divAndAssign", test_divAndAssign),
    ]
    
    //--------------------------------------------------------------------------
    // test_addSubMulDivComplex
    func test_addSubMulDivComplex() { Current.platform.test_addSubMulDivComplex() }
    func test_add() { Current.platform.test_add() }
    func test_addInt32() { Current.platform.test_addInt32() }
    func test_addUInt8() { Current.platform.test_addUInt8() }
    func test_addScalar() { Current.platform.test_addScalar() }
    func test_addAndAssign() { Current.platform.test_addAndAssign() }
    func test_subtract() { Current.platform.test_subtract() }
    func test_subtractScalar() { Current.platform.test_subtractScalar() }
    func test_subtractVector() { Current.platform.test_subtractVector() }
    func test_subtractAndAssign() { Current.platform.test_subtractAndAssign() }
    func test_mul() { Current.platform.test_mul() }
    func test_mulScalar() { Current.platform.test_mulScalar() }
    func test_mulAndAssign() { Current.platform.test_mulAndAssign() }
    func test_div() { Current.platform.test_div() }
    func test_divScalar() { Current.platform.test_divScalar() }
    func test_divAndAssign() { Current.platform.test_divAndAssign() }
}
