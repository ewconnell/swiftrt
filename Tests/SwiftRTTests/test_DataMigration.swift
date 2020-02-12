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

class test_DataMigration: XCTestCase {
    override class func setUp() {
        Current.platform = Platform<TestCpuService>()
    }
    
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_copy", test_copy),
        ("test_stressCopyOnWriteDevice", test_stressCopyOnWriteDevice),
        ("test_viewMutateOnWrite", test_viewMutateOnWrite),
        ("test_tensorDataMigration", test_tensorDataMigration),
        ("test_mutateOnDevice", test_mutateOnDevice),
        ("test_copyOnWriteDevice", test_copyOnWriteDevice),
        ("test_copyOnWriteCrossDevice", test_copyOnWriteCrossDevice),
        ("test_copyOnWrite", test_copyOnWrite),
        ("test_columnMajorDataView", test_columnMajorDataView),
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
    // test_stressCopyOnWriteDevice
    // stresses view mutation and async copies on device
    func test_stressCopyOnWriteDevice() {
        Current.log.level = .diagnostic
        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
        let matrix = Matrix(3, 2, with: 0..<6, name: "matrix")
        
        for i in 0..<500 {
            var matrix2 = matrix
            matrix2[1, 1] = 7
            let value = matrix2[1, 1]
            if value != 7.0 {
                XCTFail("i: \(i)  value is: \(value)")
                break
            }
        }
    }
    
    //==========================================================================
	// test_viewMutateOnWrite
	func test_viewMutateOnWrite() {
        Current.log.level = .diagnostic
        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
        
        // create a Matrix and give it an optional name for logging
        var m0 = Matrix(3, 4, with: 0..<12, name: "weights")
        
        XCTAssert(!m0.writeWillMutateView())
        let _ = m0.readWrite()
        
        XCTAssert(!m0.writeWillMutateView())
        let _ = m0.readOnly()
        
        XCTAssert(!m0.writeWillMutateView())
        let _ = m0.readWrite()
        
        // copy the view
        var m1 = m0
        // rw access m0 should mutate m0
        XCTAssert(m0.writeWillMutateView())
        let _ = m0.readWrite()
        
        // m1 should now be unique reference
        XCTAssert(m1.isUniqueReference())
        XCTAssert(!m1.writeWillMutateView())
        let _ = m1.readOnly()
        
        // copy the view
        var m2 = m0
        let _ = m2.readOnly()
        
        // rw request should cause copy of m0 data
        XCTAssert(m2.writeWillMutateView())
        let _ = m2.readWrite()
        // m2 should now be unique reference
        XCTAssert(m2.isUniqueReference())
	}
	
    //==========================================================================
    // test_tensorDataMigration
    //
    // This test uses the default UMA cpu queue, combined with the
    // testCpu1 and testCpu2 device queues.
    // The purpose is to test data replication and synchronization in the
    // following combinations.
    //
    // `app` means app thread
    // `uma` means any device that shares memory with the app thread
    // `discreet` is any device that does not share memory with the app thread
    // `same service` means moving data within (cuda gpu:0 -> gpu:1)
    // `cross service` means moving data between services
    //                 (cuda gpu:1 -> cpu cpu:0)
    //
    func test_tensorDataMigration() {
//        Current.log.level = .diagnostic
//        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
//
//        // create a tensor and validate migration
//        var view = Matrix(6, 4, with: 0..<24)
//
//        _ = view.readOnly()
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        _ = view.readOnly()
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        // this device is not UMA so it
//        // ALLOC device array on cpu:1
//        // COPY  cpu:0 --> cpu:1_q0
//        _ = view.readOnly(using: queue1)
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
//
//        // write access hasn't been taken, so this is still up to date
//        _ = view.readOnly()
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        // an up to date copy is already there, so won't copy
//        _ = view.readWrite(using: queue1)
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        // ALLOC device array on cpu:2
//        // COPY  cpu:1 --> cpu:2_q0
//        _ = view.readOnly(using: queue2)
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
//
//        _ = view.readOnly(using: queue1)
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        _ = view.readOnly(using: queue2)
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        _ = view.readWrite(using: queue1)
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        // the master is on cpu:1 so we need to update cpu:2's version
//        // COPY cpu:1 --> cpu:2_q0
//        _ = view.readOnly(using: queue2)
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
//
//        _ = view.readWrite(using: queue2)
//        XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)
//
//        // the master is on cpu:2 so we need to update cpu:1's version
//        // COPY cpu:2 --> cpu:1_q0
//        _ = view.readWrite(using: queue1)
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
//
//        // the master is on cpu:1 so we need to update cpu:2's version
//        // COPY cpu:1 --> cpu:2_q0
//        _ = view.readWrite(using: queue2)
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
//
//        // accessing data without a queue causes transfer to the host
//        // COPY discreet_cpu:2_q0 --> cpu:0
//        _ = view.readOnly()
//        XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
    }

    //==========================================================================
    // test_mutateOnDevice
    func test_mutateOnDevice() {
        Current.log.level = .diagnostic
        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
//
//        // create a Matrix on device 1 and fill with indexes
//        // memory is only allocated on device 1. This also shows how a
//        // temporary can be used in a scope. No memory is copied.
//        var matrix = using(device: 1) {
//            Matrix(3, 2).filled(with: 0..<6)
//        }
//
//        // retreive value on app thread
//        // memory is allocated in the host app space and the data is copied
//        // from device 1 to the host using queue 0.
//        XCTAssert(matrix[1, 1] == 3)
//
//        // simulate a readonly kernel access on device 1.
//        // matrix was not previously modified, so it is up to date
//        // and no data movement is necessary
//        _ = matrix.readOnly(using: queue1)
//
//        // sum device 1 copy, which should equal 15.
//        // This `sum` syntax creates a temporary result on device 1,
//        // then `asElement` causes the temporary to be transferred to
//        // the host, the value is retrieved, and the temp is released.
//        // This syntax is good for experiments, but should not be used
//        // for repetitive actions
//        var sum = using(device1) {
//            matrix.sum().element
//        }
//        XCTAssert(sum == 15)
//
//        // copy the matrix and simulate a readOnly operation on device2
//        // a device array is allocated on device 2 then the master copy
//        // on device 1 is copied to device 2.
//        // Since device 1 and 2 are in the same service, a device to device
//        // async copy is performed. In the case of Cuda, it would travel
//        // across nvlink and not the PCI bus
//        let matrix2 = matrix
//        _ = matrix2.readOnly(using: queue2)
//
//        // copy matrix2 and simulate a readWrite operation on device2
//        // this causes copy on write and mutate on device
//        var matrix3 = matrix2
//        _ = matrix3.readWrite(using: queue2)
//
//        // sum device 1 copy should be 15
//        // `sum` creates a temp result tensor, allocates an array on
//        // device 2, and performs the reduction.
//        // Then `asElement` causes a host array to be allocated, and the
//        // the data is copied from device 2 to host, the value is returned
//        // and the temporary tensor is released.
//        sum = using(device2) {
//            matrix.sum().element
//        }
//        XCTAssert(sum == 15.0)
//
//        // matrix is overwritten with a new array on device 1
//        matrix = using(device1) {
//            matrix.filledWithIndex()
//        }
//
//        // sum matrix on device 2
//        // `sum` creates a temporary result tensor on device 2
//        // a device array for `matrix` is allocated on device 2 and
//        // the matrix data is copied from device 1 to device 2
//        // then `asElement` creates a host array and the result is
//        // copied from device 2 to the host array, and then the tensor
//        // is released.
//        sum = using(device2) {
//            matrix.sum().element
//        }
//        XCTAssert(sum == 15.0)
    }
    
    //--------------------------------------------------------------------------
    // test_copyOnWriteDevice
    func test_copyOnWriteDevice() {
//        Current.log.level = .diagnostic
//        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

//        // fill with index on device 1
//        var matrix1 = Matrix(3, 2)
//        using(device: 1) {
//            fillWithIndex(&matrix1)
//        }
//        // testing a value causes the data to be copied to the host
//        XCTAssert(matrix1[1, 1] == 3.0)
//
//        // copy and mutate data
//        // the data will be duplicated wherever the source is
//        var matrix2 = matrix1
//        XCTAssert(matrix2[1, 1] == 3.0)
//
//        // writing to matrix2 causes view mutation and copy on write
//        matrix2[1, 1] = 7
//        XCTAssert(matrix1[1, 1] == 3.0)
//
//        XCTAssert(matrix2[1, 1] == 7.0)
    }
    
    //--------------------------------------------------------------------------
    // test_copyOnWriteCrossDevice
    func test_copyOnWriteCrossDevice() {
//            Current.log.level = .diagnostic
//            Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
//        var matrix1 = Matrix(3, 2)
        
//        // allocate array on device 1 and fill with indexes
//        using(device: 1) {
//            fillWithIndex(&matrix1)
//        }
//
//        // getting a value causes the data to be copied to an
//        // array associated with the app thread
//        // The master version is stil on device 1
//        let value = matrix1[1, 1]
//        XCTAssert(value == 3)
//
//        // simulate read only access on device 1 and 2
//        // data will be copied to device 2 for the first time
//        _ = matrix1.readOnly(using: queue1)
//        _ = matrix1.readOnly(using: queue2)
//
//        // sum device 1 copy should be 15
//        let sum1 = using(device: 1) {
//            matrix1.sum().element
//        }
//        XCTAssert(sum1 == 15)
//
//        // clear the device 0 master copy
//        using(device: 1) {
//            fill(&matrix1, with: 0)
//        }
//
//        // sum device 1 copy should now also be 0
//        // sum device 1 copy should be 15
//        let sum2 = using(device: 2) {
//            matrix1.sum().element
//        }
//        XCTAssert(sum2 == 0)
    }

    //--------------------------------------------------------------------------
    // test_copyOnWrite
    // NOTE: uses the default queue
    func test_copyOnWrite() {
//        Current.log.level = .diagnostic
//        Current.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
        
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
        let expected = [Int32](0..<6)
        let values = cmMatrix.flatArray
        XCTAssert(values == expected, "values don't match")
    }
}
