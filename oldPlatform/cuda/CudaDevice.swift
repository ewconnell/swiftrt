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
import CCuda

public class CudaDevice : LocalComputeDevice {
    //--------------------------------------------------------------------------
    // properties
    public private(set) var trackingId = 0
    public private(set) weak var service: ComputeService!
    public private(set) var queues = [DeviceQueue]()
    public let attributes: [String : String]
    public let deviceArrayReplicaKey = Platform.nextDeviceArrayReplicaKey
    public let id: Int
    public var logInfo: LogInfo
    public var maxThreadsPerBlock: Int { return Int(props.maxThreadsPerBlock) }
    public let name: String
    public var timeout: TimeInterval?
    public var memory: MemoryProperties
    public var limits: DeviceLimits
    public var utilization: Float = 0
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()
    
    private let props: cudaDeviceProp

    //--------------------------------------------------------------------------
	// initializers
    public init(service: CudaService,
                deviceId: Int,
                logInfo: LogInfo,
                isUnified: Bool,
                timeout: TimeInterval?) throws
    {
        self.logInfo = logInfo.flat("gpu:\(deviceId)")
        self.id = deviceId
        self.service = service
        self.timeout = timeout

        //---------------------------------
        // get device properties
        var tempProps = cudaDeviceProp()
        try cudaCheck(status: cudaGetDeviceProperties(&tempProps, 0))
        props = tempProps
        let nameCapacity = MemoryLayout.size(ofValue: tempProps.name)

        name = withUnsafePointer(to: &tempProps.name) {
            $0.withMemoryRebound(to: UInt8.self, capacity: nameCapacity) {
                String(cString: $0)
            }
        }

        // initialize attribute list
        attributes = [
            "name"               : self.name,
            "compute capability" : "\(props.major).\(props.minor)",
            "global memory"      : "\(props.totalGlobalMem / (1024 * 1024)) MB",
            "multiprocessors"    : "\(props.multiProcessorCount)",
        ]
        
        // TODO: determine meaningful values, not currently used
        self.limits = DeviceLimits(
            maxComputeSharedMemorySize: 1,
            maxComputeWorkGroupCount: (1, 1, 1),
            maxComputeWorkGroupInvocations: 1,
            maxComputeWorkGroupSize: (1, 1, 1),
            maxMemoryAllocationCount: 1
        )

        // TODO:
        self.memory = MemoryProperties(addressing: .discreet,
                                       heaps: [MemoryHeap]())

        //---------------------------------
        // create device queues
        assert(Platform.cudaConfiguration[.queuesPerDevice] is Int)
        let queueCount = Platform.cudaConfiguration[.queuesPerDevice] as! Int
        var queues = [DeviceQueue]()
        for queueId in 0..<queueCount {
            let queueName = "queue:\(queueId)"
            try queues.append(CudaQueue(logInfo: logInfo.flat(queueName),
                                        device: self, name: queueName,
                                        id: queueId))
        }
        self.queues = queues
        
        //---------------------------------
        // devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
                .register(self, namePath: logNamePath, isStatic: true)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }

	//--------------------------------------------------------------------------
	// createArray
    public func createArray(byteCount: Int, heapIndex: Int, zero: Bool) throws
        -> DeviceArray
    {
        return try CudaDeviceArray(device: self, byteCount: byteCount,
                                   zero: zero)
	}

    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a device array from a uma buffer.
    public func createMutableReferenceArray(
            buffer: UnsafeMutableRawBufferPointer) -> DeviceArray {
        return CudaDeviceArray(device: self, buffer: buffer)
    }

    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a device array from a uma buffer.
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
                    -> DeviceArray {
        return CudaDeviceArray(device: self, buffer: buffer)
    }

	//--------------------------------------------------------------------------
	// select
	public func select() throws {
		try cudaCheck(status: cudaSetDevice(Int32(id)))
	}
}
