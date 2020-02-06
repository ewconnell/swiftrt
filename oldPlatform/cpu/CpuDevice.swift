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
import Foundation
import Dispatch

//==============================================================================
/// CpuQueueProtocol
public protocol CpuQueueProtocol: DeviceQueue {
    init(logInfo: LogInfo, device: ComputeDevice, name: String, id: Int)
}

//==============================================================================
/// CpuDevice
public final class CpuDevice<QueueT>: LocalComputeDevice
    where QueueT: CpuQueueProtocol
{
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public private(set) var queues = [DeviceQueue]()
    public let deviceArrayReplicaKey = Platform.nextDeviceArrayReplicaKey
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    public private(set) weak var service: ComputeService!
    public var timeout: TimeInterval?
    public var deviceErrorHandler: DeviceErrorHandler?
    public var limits: DeviceLimits
    public var memory: MemoryProperties
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()
    
    //--------------------------------------------------------------------------
	// initializers
	public init(service: ComputeService,
                deviceId: Int,
                logInfo: LogInfo,
                addressing: MemoryAddressing,
                timeout: TimeInterval?)
    {
        self.name = addressing == .unified ?
            "cpu:\(deviceId)" : "discreet_cpu:\(deviceId)"
		self.logInfo = logInfo.flat(self.name)
		self.id = deviceId
		self.service = service
        self.timeout = timeout
        
        // TODO: determine meaningful values, not currently used
        self.limits = DeviceLimits(
            maxComputeSharedMemorySize: 1,
            maxComputeWorkGroupCount: (1, 1, 1),
            maxComputeWorkGroupInvocations: 1,
            maxComputeWorkGroupSize: (1, 1, 1),
            maxMemoryAllocationCount: 1
        )
        
        // TODO:
        self.memory = MemoryProperties(addressing: addressing,
                                       heaps: [MemoryHeap]())

        //---------------------------------
        // create device queues
        assert(Platform.cpuConfiguration[.queuesPerDevice] is Int)
        let queueCount = Platform.cpuConfiguration[.queuesPerDevice] as! Int
        var queues = [DeviceQueue]()
        for queueId in 0..<queueCount {
            let queueName = "queue:\(queueId)"
            queues.append(QueueT(logInfo: logInfo.flat(queueName),
                                 device: self, name: queueName,
                                 id: queueId))
        }
        self.queues = queues

		// devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
            .register(self, namePath: logNamePath, isStatic: true)
	}

    @inlinable
	deinit { ObjectTracker.global.remove(trackingId: trackingId) }

    //--------------------------------------------------------------------------
	// createArray
	//	This creates memory on the device
    @inlinable
    public func createArray(byteCount: Int, heapIndex: Int, zero: Bool) throws
        -> DeviceArray
    {
        CpuDeviceArray(device: self, byteCount: byteCount, zero: zero)
	}
    
    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a device array from a uma buffer.
    @inlinable
    public func createMutableReferenceArray(
        buffer: UnsafeMutableRawBufferPointer) -> DeviceArray {
        CpuDeviceArray(device: self, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a device array from a uma buffer.
    @inlinable
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
        -> DeviceArray
    {
        CpuDeviceArray(device: self, buffer: buffer)
    }
}
