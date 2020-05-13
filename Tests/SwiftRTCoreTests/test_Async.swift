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
        ("test_add", test_add),
    ]
    
    // append and use a discreet async cpu device for these tests
    override func setUpWithError() throws {
        Context.log.level = .diagnostic
        // append a cpu device
        let asyncDiscreetCpu = Context.devices.count
        let logInfo = Context.local.platform.logInfo
        let testDevice = CpuDevice(parent: logInfo, memoryType: .discreet,
                                   id: asyncDiscreetCpu, queueMode: .async)
        Context.local.platform.devices.append(testDevice)
        use(device: asyncDiscreetCpu)
    }
    
    override func tearDownWithError() throws {
        Context.local.platform.devices.removeLast()
        use(device: 0)
        Context.log.level = .error
    }

    //--------------------------------------------------------------------------
    func test_add() {
        let a = array(0..<6)
        let b = array(0..<6)
        let c = a + b
        
        // sync with caller
        let result = c.array
        print(result)
        XCTAssert(result == [0, 2, 4, 6, 8, 10])
    }
}
