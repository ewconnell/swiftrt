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
/// MultiDeviceBuffer
/// Used internally to manage the state of a collection of device buffers
public final class MultiDeviceBuffer<Element>: ElementBuffer {
    public var id: Int
    public var isReference: Bool
    
    /// the number of elements in the buffer
    public let count: Int
    /// a dictionary of device memory replicas allocated on each device
    /// - Parameter key: the device index
    /// - Returns: the associated device memory object
    public var memory: [Int : DeviceMemory]
    
    /// `true` if the buffer is not mutable, such as in the case of a readOnly
    /// reference buffer.
    public let isReadOnly: Bool
    
    /// the `id` of the last queue that obtained write access
    public var lastMutatingQueue: QueueId
    
    /// the buffer name used in diagnostic messages
    public let name: String
    
    /// the index of the device holding the master version
    public var masterDevice: Int
    
    /// the masterVersion is incremented each time write access is taken.
    /// All device buffers will stay in sync with this version, copying as
    /// necessary.
    public var masterVersion: Int
    
    //--------------------------------------------------------------------------
    // init(type:count:name:
    @inlinable
    public init(count: Int, name: String) {
        self.count = count
        self.memory = [Int : DeviceMemory]()
        self.id = Platform.nextBufferId
        self.isReference = false
        self.isReadOnly = false
        self.lastMutatingQueue = QueueId(0, 0)
        self.masterDevice = 0
        self.masterVersion = 0
        self.name = name
    }
    
    //--------------------------------------------------------------------------
    /// `deallocate`
    /// releases device memory associated with this buffer descriptor
    /// - Parameter device: the device to release memory from. `nil` will
    /// release all associated device memory for this buffer.
    @inlinable
    public func deallocate(device: Int? = nil) {
        if let device = device {
            memory[device]!.deallocate()
        } else {
            memory.values.forEach { $0.deallocate() }
        }
    }

    @inlinable
    public init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String) {
        fatalError()
    }
    
    @inlinable
    public init(referenceTo buffer: UnsafeMutableBufferPointer<Element>, name: String) {
        fatalError()
    }
    
    @inlinable
    public init<C>(elements: C, name: String) where C : Collection, C.Element == Element {
        fatalError()
    }
    
    @inlinable
    public init<Shape, Stream>(block shape: Shape, bufferedBlocks: Int, stream: Stream) where Shape : ShapeProtocol, Stream : BufferStream {
        fatalError()
    }
    
    @inlinable
    public init(for element: Element) {
        fatalError()
    }
    
    @inlinable
    public func duplicate() -> Self {
        fatalError()
    }
    
    @inlinable
    public func read(at offset: Int, count: Int, using queue: DeviceQueue) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    @inlinable
    public func readWrite(at offset: Int, count: Int, willOverwrite: Bool, using queue: DeviceQueue) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
}

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
    @inlinable
    public init() {
        self.name = "CudaService"
        self.logInfo = LogInfo(logWriter: Platform.log,
                               logLevel: .error,
                               namePath: self.name,
                               nestingLevel: 0)
        self.devices = [
            CudaDevice(parent: logInfo, id: 0),
            CudaDevice(parent: logInfo, id: 1)
        ]
        
        // select device 0 queue 0 by default
        self.queueStack = []
        self.queueStack = [ensureValidId(1, 0)]
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

    @inlinable
    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.id = id
        self.name = "gpu:\(id)"
        self.logInfo = parentLogInfo.child(name)

        // create queues
        let isCpuDevice = id == 0
        self.memoryType = isCpuDevice ? .unified : .discreet
        let numQueues = isCpuDevice ? 1 : 3
        self.queues = []
        for queueId in 0..<numQueues {
            queues.append(CudaQueue(id: queueId, parent: logInfo,
                                    deviceId: self.id, deviceName: name,
                                    useGpu: !isCpuDevice))
        }
    }

    @inlinable
    public func allocate(byteCount: Int, heapIndex: Int) -> DeviceMemory {
        // TODO
        let buffer = UnsafeMutableRawBufferPointer(start: nil, count: byteCount)
        return DeviceMemory(buffer: buffer, memoryType: memoryType, {})
    }
}

//==============================================================================
/// CudaQueue
public struct CudaQueue: DeviceQueue, DeviceFunctions {
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
        self.id = id
        self.name = "q:\(id)"
        self.logInfo = logInfo.child(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.useGpu = useGpu
        
        diagnostic("\(createString) DeviceQueue " +
            "\(deviceName)_\(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------
    // protocol functions
    @inlinable
    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        fatalError()
    }
    
    @inlinable
    public func record(event: QueueEvent) -> QueueEvent {
        fatalError()
    }
    
    @inlinable
    public func wait(for event: QueueEvent) {
        fatalError()
    }
    
    @inlinable
    public func waitUntilQueueIsComplete() {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // copyAsync
    @inlinable
    public func copyAsync(from deviceMemory: DeviceMemory,
                          to otherDeviceMemory: DeviceMemory)
    {
        assert(deviceMemory.memoryType == .unified &&
            otherDeviceMemory.memoryType == .unified)
        
        let buffer = UnsafeRawBufferPointer(deviceMemory.buffer)
        otherDeviceMemory.buffer.copyMemory(from: buffer)
    }
}
