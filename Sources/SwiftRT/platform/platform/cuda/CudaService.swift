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
/// CudaService
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public struct CudaService: ComputeService {
    // properties
    public let devices: [CudaDevice]
    public let id: Int
    public let logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.name = "Cuda"
        self.logInfo = parentLogInfo.child(name)
        self.id = id
        self.devices = [
            CudaDevice(parent: logInfo, id: 0),
            CudaDevice(parent: logInfo, id: 1)
        ]
    }
}

//==============================================================================
/// CudaDevice
public struct CudaDevice: ComputeDevice {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let name: String
    public var queues: [CudaQueue]

    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.id = id
        self.name = "gpu:\(id)"
        self.logInfo = parentLogInfo.child(name)

        // create queues
        let isCpuDevice = id == 0
        let numQueues = isCpuDevice ? 1 : 3
        self.queues = []
        for queueId in 0..<numQueues {
            queues.append(CudaQueue(parent: logInfo, deviceName: name,
                                    id: queueId, useGpu: !isCpuDevice))
        }
    }
}

//==============================================================================
/// CudaQueue
public struct CudaQueue: DeviceQueue, DeviceFunctions {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let deviceName: String
    public let name: String
    public let useGpu: Bool
    
    /// used to detect accidental queue access by other threads
    @usableFromInline
    let creatorThread: Thread
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(parent parentLogInfo: LogInfo, deviceName: String,
                id: Int, useGpu: Bool)
    {
        self.id = id
        self.name = "q:\(id)"
        self.logInfo = parentLogInfo.child(name)
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.useGpu = useGpu
        
        diagnostic("\(createString) DeviceQueue " +
            "\(deviceName)_\(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------
    // protocol functions
    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        CpuQueueEvent()
    }
    
    public func record(event: QueueEvent) -> QueueEvent {
        CpuQueueEvent()
    }
    
    public func wait(for event: QueueEvent) {
        fatalError()
    }
    
    public func waitUntilQueueIsComplete() {
        fatalError()
    }
}

//==============================================================================
/// CudaQueue
public extension CudaQueue {
    
    @inlinable
    func add<T>(_ lhs: T, _ rhs: T, _ result: inout T)
        where T: TensorView & BinaryInteger
    {
        guard useGpu else { cpu_add(lhs, rhs, &result); return }
        
        // gpu version here
        result = lhs + rhs
    }
}
