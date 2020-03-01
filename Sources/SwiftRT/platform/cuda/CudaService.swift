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
public class CudaService: PlatformService {
    // properties
    public let devices: [CudaDevice]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [QueueId]

    //--------------------------------------------------------------------------
    public init() {
        name = "Cuda"
        logInfo = LogInfo(logWriter: Platform.log, logLevel: .error,
                          namePath: name, nestingLevel: 0)
        
        // add a device whose queue is synchronized with the application
        var installedDevices = [CudaDevice(parent: logInfo, id: 0)]

        // query for installed cuda devices and add
        installedDevices.append(CudaDevice(parent: logInfo, id: 1))
        
        devices = installedDevices
        
        // select device 1 queue 0 by default
        queueStack = []
        queueStack = [ensureValidId(1, 0)]
    }
}

//==============================================================================
/// CudaDevice
public struct CudaDevice: ServiceDevice {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    public var queues: [CudaQueue]

    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.id = id
        self.name = "gpu:\(id)"
        self.logInfo = parentLogInfo.child(name)

        // create queues
        let isCpuDevice = id == 0
        var numQueues: Int
        if isCpuDevice {
            memoryType = .unified
            numQueues = 1
        } else {
            memoryType = .discreet
            numQueues = 3
        }

        self.queues = []
        for i in 0..<numQueues {
            queues.append(CudaQueue(id: i, parent: logInfo,
                                    deviceId: self.id, deviceName: name,
                                    useGpu: !isCpuDevice))
        }
    }
    
    public func allocate(byteCount: Int, heapIndex: Int) -> DeviceMemory {
        fatalError()
    }
}

//==============================================================================
/// CudaQueue
public struct CudaQueue: DeviceQueue {
    // properties
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceId: Int
    public let deviceName: String
    public let id: Int
    public let logInfo: LogInfo
    public let name: String
    public let useGpu: Bool
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                deviceId: Int, deviceName: String, useGpu: Bool)
    {
        self.useGpu = useGpu
        self.id = id
        self.name = "queue:\(id)"
        self.logInfo = logInfo.flat(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()

        diagnostic("\(createString) \(Self.self): \(deviceName)_\(name)",
            categories: .queueAlloc)
    }

    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        fatalError()
    }
    
    public func record(event: QueueEvent) -> QueueEvent {
        fatalError()
    }
    
    public func wait(for event: QueueEvent) {
        fatalError()
    }
    
    public func waitUntilQueueIsComplete() {
        fatalError()
    }
    
    public func copyAsync(from deviceMemory: DeviceMemory,
                          to otherDeviceMemory: DeviceMemory)
    {
        fatalError()
    }
}

//==============================================================================
/// CudaQueue
public extension CudaQueue {
    
//    @inlinable
//    func add<T>(_ lhs: T, _ rhs: T, _ result: inout T)
//        where T: TensorView & BinaryInteger
//    {
//        guard useGpu else { cpu_add(lhs, rhs, &result); return }
//
//        // gpu version here
//        result = lhs + rhs
//    }
}
