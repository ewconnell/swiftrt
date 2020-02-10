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
/// CpuService
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public struct CpuService: ComputeService {
    // properties
    public let devices: [CpuDevice]
    public let id: Int
    public let logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    @inlinable
    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.name = "CpuService"
        self.logInfo = parentLogInfo.child(name)
        self.id = id
        self.devices = [CpuDevice(parent: logInfo, id: 0)]
    }
}

//==============================================================================
/// CpuQueue
public struct CpuQueue: DeviceQueue, Logger {
    // properties
    public let arrayReplicaKey: Int
    public let defaultQueueEventOptions: QueueEventOptions
    public let deviceId: Int
    public let id: Int
    public let logInfo: LogInfo
    public let deviceName: String
    public let memoryAddressing: MemoryAddressing
    public let name: String

    /// used to detect accidental queue access by other threads
    @usableFromInline
    let creatorThread: Thread

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                replicationKey: Int,
                deviceId: Int, deviceName: String)
    {
        self.arrayReplicaKey = replicationKey
        self.id = id
        self.name = "q:\(id)"
        self.logInfo = logInfo.child(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.memoryAddressing = .unified

        diagnostic("\(createString) DeviceQueue " +
            "\(deviceName)_\(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------

    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        CpuQueueEvent()
    }
    
    public func record(event: QueueEvent) -> QueueEvent {
        CpuQueueEvent()
    }
    
    public func wait(for event: QueueEvent) {
    }
    
    public func waitUntilQueueIsComplete() {
    }
    
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
    }
    
    public func copyAsync(to array: DeviceArray, from otherArray: DeviceArray) {
    }
    
    public func copyAsync(to array: DeviceArray, from hostBuffer: UnsafeRawBufferPointer) {
    }
    
    public func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer, from array: DeviceArray) {
    }
    
    public func zero(array: DeviceArray) {
    }
}

//==============================================================================
/// CpuQueueEvent
public struct CpuQueueEvent: QueueEvent {
    // properties
    public let id: Int
    public var occurred: Bool
    public var recordedTime: Date?

    //--------------------------------------------------------------------------
    public init() {
        // the queue is synchronous so the event has already occurred
        self.id = 0
        self.occurred = true
    }
    
    @inlinable
    public func wait() throws {
    }
}
