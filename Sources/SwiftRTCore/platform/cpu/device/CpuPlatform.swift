//******************************************************************************
// Copyright 2020 Google LLC
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
import Foundation

//==============================================================================
/// CpuPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public final class CpuPlatform: ComputePlatform {
    // types
    public typealias Storage = CpuStorage

    // shared
    public let acceleratorQueueCount: Int = 0
    public static let startTime = Date()
    public static var cpuQueueCount =
        ProcessInfo.processInfo.activeProcessorCount
    public static var discreteMemoryDeviceId: Int { 1 }
    public static var objectId = AtomicCounter()
    public static var queueId = AtomicCounter()
    public static var eventId = AtomicCounter()
    public static var _randomSeed: RandomSeed = generateRandomSeed()
    public static let syncQueue =
        CpuQueue(deviceIndex: 0, name: "appThread",
                 queueMode: .sync, memoryType: .unified)

    // properties
    public var devices: [CpuDevice]
    public var evaluationModeStack: [EvaluationMode]
    public let name: String
    public var queueStack: [CpuQueue]
    @inlinable public var currentQueue: CpuQueue { queueStack.last! }
    @inlinable public var currentDevice: CpuDevice {
        devices[currentQueue.deviceIndex]
    }

    // thread local instance
    @usableFromInline static let mainPlatform = CpuPlatform()
    @inlinable public static var local: CpuPlatform { Self.mainPlatform }

    //-------------------------------------
    // HACK for AD
    /// a storage buffer with a single zero value which is shared
    /// every time Element.zero is obtained by AD.
    // used to minimize AD overhead. AD needs to fix this problem.
    public static let zeroStorage: Storage = {
        Storage(storedElement: Int64(0), name: "Zero")
    }()

    //--------------------------------------------------------------------------
    @inlinable public init() {
        name = "\(Self.self)"
        evaluationModeStack = [.inferring]
        
        // create the device and default number of async queues
        let device = CpuDevice(index: 0, memoryType: .unified,
                               queueCount: Platform.cpuQueueCount)
        
        // create a discrete memory unit test device
        let test = CpuDevice(index: 1, memoryType: .discrete, queueCount: 2)
        devices = [device, test]

        // make the app thread queue current by default
        queueStack = [Self.syncQueue]

        diagnostic(.device, "create  sync queue: appThread",
                   categories: .queueAlloc)
    }
}
