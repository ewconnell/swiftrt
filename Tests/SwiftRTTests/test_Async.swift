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

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_hostMultiWrite", test_hostMultiWrite),
        ("test_defaultQueueOp", test_defaultQueueOp),
        ("test_secondaryDiscreetMemoryQueue", test_secondaryDiscreetMemoryQueue),
        ("test_threeQueueInterleave", test_threeQueueInterleave),
        ("test_tensorReferenceBufferSync", test_tensorReferenceBufferSync),
        ("test_QueueEventWait", test_QueueEventWait),
        ("test_perfCreateQueueEvent", test_perfCreateQueueEvent),
        ("test_perfRecordQueueEvent", test_perfRecordQueueEvent),
    ]

    //==========================================================================
    // test_hostMultiWrite
    // accesses a tensor on the host by dividing the first dimension
    // into batches and concurrently executing a user closure for each batch
    func test_hostMultiWrite() {
        do {
//            Platform.log.level = .diagnostic
            typealias Pixel = RGB<UInt8>
            typealias ImageSet = VolumeType<Pixel>
            let expected = Pixel(0, 127, 255)
            var trainingSet = ImageSet(extents: (20, 256, 256))

            try trainingSet.hostMultiWrite(synchronous: true) { batch in
                for i in 0..<batch.items {
                    // at this point load image data from a file or database,
                    // decompress, type convert, whatever is needed
                    // In this example we'll just fill the buffer with
                    // the `expected` value
                    // the buffer is already in host memory so it can't fail
                    batch[i].readWrite().initialize(repeating: expected)
                }
            }

            // check the last item to see if it contains the expected value
            let lastItem = trainingSet[-1]
            XCTAssert(lastItem.first == expected)
        } catch {
            XCTFail(String(describing: error))
        }
        
        // check for object leaks
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_defaultQueueOp
    // initializes two matrices and adds them together
    func test_defaultQueueOp() {
//        Platform.log.level = .diagnostic
        do {
            let m1 = IndexMatrix(2, 5, with: 0..<10, name: "m1")
            let m2 = IndexMatrix(2, 5, with: 0..<10, name: "m2")
            let result = m1 + m2
            XCTAssert(result == (0..<10).map { $0 + $0 })
        }
        
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_secondaryDiscreetMemoryQueue
    // initializes two matrices on the app thread, executes them on `queue1`,
    // the retrieves the results
    func test_secondaryDiscreetMemoryQueue() {
//        Platform.log.level = .diagnostic
//        Platform.log.categories = [.dataAlloc, .dataCopy, .scheduling, .queueSync]
        
        do {
            let device1 = Platform.testCpu1
            
            let m1 = IndexMatrix(2, 5, with: 0..<10, name: "m1")
            let m2 = IndexMatrix(2, 5, with: 0..<10, name: "m2")
            
            // perform on user provided discreet memory queue
            // synchronize with host queue and retrieve result values
            let result = using(device1) { m1 + m2 }
            XCTAssert(result == (0..<10).map { $0 + $0 })
        }
        
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_threeQueueInterleave
    func test_threeQueueInterleave() {
//        Platform.log.level = .diagnostic
        do {
            let device1 = Platform.testCpu1
            let device2 = Platform.testCpu2
            
            let m1 = IndexMatrix(2, 3, with: 0..<6, name: "m1")
            let m2 = IndexMatrix(2, 3, with: 0..<6, name: "m2")
            let m3 = IndexMatrix(2, 3, with: 0..<6, name: "m3")
            
            // sum the values with a delay on device 1
            let sum_m1m2: IndexMatrix = using(device1) {
                DeviceContext.currentQueue.delayQueue(atLeast: 0.1)
                return m1 + m2
            }
            
            // multiply the values on device 2
            let result = using(device2) {
                sum_m1m2 * m3
            }
            XCTAssert(result == (0..<6).map { ($0 + $0) * $0 })
        }
        
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_tensorReferenceBufferSync
    func test_tensorReferenceBufferSync() {
    }

    //==========================================================================
    // test_QueueEventWait
    func test_QueueEventWait() {
//        Platform.log.level = .diagnostic
//        Platform.log.categories = [.queueSync]
        
        do {
            let queue = Platform.testCpu1.queues[0]
            let event = try queue.createEvent()
            queue.delayQueue(atLeast: 0.001)
            try queue.record(event: event).wait()
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_perfCreateQueueEvent
    // measures the event overhead of creating 10,000 events
    func test_perfCreateQueueEvent() {
        #if !DEBUG
        let queue = Platform.testCpu1.queues[0]
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try queue.createEvent()
                }
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }

    //==========================================================================
    // test_perfRecordQueueEvent
    // measures the event overhead of processing 10,000 tensors
    func test_perfRecordQueueEvent() {
        #if !DEBUG
        let queue = Platform.testCpu1.queues[0]
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try queue.record(event: queue.createEvent())
                }
                try queue.waitUntilQueueIsComplete()
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }
}
