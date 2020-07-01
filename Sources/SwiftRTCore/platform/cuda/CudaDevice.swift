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
import CCuda

//==============================================================================
/// CudaDevice
public class CudaDevice: ComputeDevice {
    // properties
    public let id: Int
    public let gpuId: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    public var queues: [CudaQueue]
    public var properties: [CudaDeviceProperties : String]

    @inlinable public init(
        id: Int,
        gpuId: Int,
        parent logInfo: LogInfo
    ) {
        self.id = id
        name = id == 0 ? "cpu:\(id)" : "gpu:\(gpuId)"
        memoryType = id == 0 ? .unified : .discrete
        self.logInfo = logInfo.flat(name)
        self.gpuId = gpuId
        self.properties = [:]
        queues = []

        if id == 0 {
            // cpu device case
            for _ in 0..<Context.cpuQueueCount {
                let queueId = Context.nextQueueId
                queues.append(CudaQueue(id: queueId,
                                        parent: logInfo,
                                        gpuId: 0,
                                        name: "\(name)_q\(queueId)",
                                        mode: .async,
                                        useGpu: false))
            }
        } else {
            // gpu device case
            for _ in 0..<Context.acceleratorQueueCount {
                let queueId = Context.nextQueueId
                queues.append(CudaQueue(id: queueId,
                                        parent: logInfo,
                                        gpuId: gpuId,
                                        name: "\(name)_q\(queueId)",
                                        mode: .async,
                                        useGpu: true))
            }
            getCudaDeviceProperties()
        }
    }

    //--------------------------------------------------------------------------
    @inlinable func getCudaDeviceProperties() {
        do {
            // get device properties
            var props = cudaDeviceProp()
            try cudaCheck(status: cudaGetDeviceProperties(&props, 0))
            let nameCapacity = MemoryLayout.size(ofValue: props.name)
            let deviceName = withUnsafePointer(to: &props.name) {
                $0.withMemoryRebound(to: UInt8.self, capacity: nameCapacity) {
                    String(cString: $0)
                }
            }
            properties = [
                .deviceName : deviceName,
                .computeCapability : "\(props.major).\(props.minor)",
                .globalMemory : "\(props.totalGlobalMem / (1024 * 1024)) MB",
                .multiprocessors : "\(props.multiProcessorCount)",
                .unifiedAddressing : "\(memoryType == .unified ? "yes" : "no")",
            ]
		} catch {
            writeLog("Failed to read device properites. \(error)")
        }
    }
}

//==============================================================================
// CudaDeviceProperties
public enum CudaDeviceProperties: Int {
    case deviceName, globalMemory, computeCapability, multiprocessors, unifiedAddressing
}

