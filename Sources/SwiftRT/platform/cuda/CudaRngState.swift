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
import Foundation
import CCuda
//import CudaKernels

public final class CudaRandomNumberGeneratorState:
    RandomNumberGeneratorState, Logging
{
    // properties
    public let seedCount: Int = 0
//    public private(set) var handle: UnsafeMutableRawPointer
//    public private(set) var stream: DeviceStream
//
//    // logging
//    public var name: String
//    public var logLevel = LogLevel.error
//    public var nestingLevel = 0
//    public weak var currentLog: Log?
//
//    // initializers
//    public init(log: Log?, using stream: DeviceStream, count: Int,
//                seed: UInt?, name: String) throws {
//        assert(stream is CudaStream)
//        self.stream = stream
//        self.count  = count
//        self.name   = name
//        currentLog  = log
//
//        // select the device
//        try stream.device.select()
//
//        // seed
//        let generatorSeed: UInt64 = seed != nil ?
//            UInt64(seed!) : UInt64(Date().timeIntervalSince1970)
//
//        // create the initial state
//        var temp: UnsafeMutableRawPointer?
//        try cudaCheck(status: cudaCreateRandomGeneratorState(
//            &temp,
//            generatorSeed,
//            count,
//            (stream as! CudaStream).handle))
//
//        handle = temp!
//
//        // log
//        trackingId = objectTracker.register(type: self)
//        if willLog(level: .diagnostic) {
//            diagnostic("\(createString) \(name)", categories: .dataAlloc)
//        }
//    }
//
//    // deinit
//    deinit {
//        do {
//            // select the device
//            try stream.device.select()
//
//            // make sure pending queued commands complete before destroying the queue
//            try stream.blockCallerUntilComplete()
//
//            // free the memory
//            cudaFree(handle);
//        } catch {
//            writeLog(String(describing: error))
//        }
//
//        if willLog(level: .diagnostic) == true {
//            diagnostic("\(releaseString) \(name)", categories: .dataAlloc)
//        }
//    }
}
