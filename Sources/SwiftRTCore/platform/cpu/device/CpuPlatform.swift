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
    public static let acceleratorQueueCount: Int = 0
    public static var cpuQueueCount = 0
    public static var discreteMemoryDeviceId: Int { 1 }
    public static var eventId = AtomicCounter()
    public static let local = CpuPlatform()
    public static let mainThread = pthread_self()
    public static var objectId = AtomicCounter()
    public static var queueId = AtomicCounter()
    public static let startTime = Date()
    public static var lastRandomSeed: RandomSeed = generateRandomSeed()

    //-------------------------------------
    public static let testDevice =
        CpuDevice(index: 1, memoryType: .discrete, queueCount: 2)

    //-------------------------------------
    // for synchrnous execution and syncing with the app thread
    public static let syncQueue =
        CpuQueue(deviceIndex: 0, name: "appThread",
                 queueMode: .sync, memoryType: .unified)

    // properties
    public var devices: [CpuDevice]
    @inlinable public var name: String { "\(Self.self)" }
    public var queueStack: [CpuQueue]

    //-------------------------------------
    // HACK for AD
    /// a storage buffer with a single zero value which is shared
    /// every time Element.zero is obtained by AD.
    // used to minimize AD overhead. AD needs to fix this problem.
    public static let zeroStorage: Storage = {
        Storage(storedElement: Int64(0), name: "Zero")
    }()

    //--------------------------------------------------------------------------
    @inlinable public init(queueCount: Int = CpuPlatform.cpuQueueCount) {
        // create the device and default number of async queues
        devices = [CpuDevice(index: 0, memoryType: .unified,
                             queueCount: queueCount)]

        // make the app thread queue current by default
        queueStack = [Self.syncQueue]

        diagnostic(.device, "create  sync queue: appThread",
                   categories: .queueAlloc)
    }
}
