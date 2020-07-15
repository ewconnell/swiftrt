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
    public let index: Int
    public let memoryType: MemoryType
    public let name: String
    public var queues: [CudaQueue]
    public var properties: [String]

    @inlinable public init(index: Int) {
        let isCpu = index == 0
        self.index = index
        self.name = "dev:\(index)"
        self.memoryType = isCpu ? .unified : .discrete
        properties = []
        queues = []

        //----------------------------
        // report device stats
        diagnostic("\(deviceString) \(name)", categories: .device)
        properties = isCpu ? getCpuProperties() : getCudaDeviceProperties()
        properties.forEach {
            diagnostic(" \($0)", categories: .device)
        }

        //----------------------------
        // create queues
        if isCpu {
            // cpu device case
            for i in 0..<Context.cpuQueueCount {
                queues.append(CudaQueue(deviceIndex: index,
                                        name: "\(name)_q\(i)",
                                        queueMode: .async,
                                        useGpu: false))
            }
        } else {
            // gpu device case
            for i in 0..<Context.acceleratorQueueCount {
                queues.append(CudaQueue(deviceIndex: index,
                                        name: "\(name)_q\(i)",
                                        queueMode: .async,
                                        useGpu: true))
            }
        }
    }

    //--------------------------------------------------------------------------
    // note: physicalMemory is being under reported on Ubuntu by ~750MB, so
    // we add one
    @inlinable func getCpuProperties() -> [String] {
        let info = ProcessInfo.processInfo
        let oneGB = UInt64(1.GB)
        return [
            "device type       : cpu",
            "physical memory   : \(info.physicalMemory / oneGB + 1) GB",
            "active cores      : \(info.activeProcessorCount)",
            "memory addressing : \(memoryType)",
        ]
    }

    //--------------------------------------------------------------------------
    @inlinable func getCudaDeviceProperties() -> [String] {
        // get device properties
        var props = cudaDeviceProp()
        cudaCheck(cudaGetDeviceProperties(&props, 0))
        let nameCapacity = MemoryLayout.size(ofValue: props.name)
        let deviceName = withUnsafePointer(to: &props.name) {
            $0.withMemoryRebound(to: UInt8.self, capacity: nameCapacity) {
                String(cString: $0)
            }
        }
        return [
            "device type       : \(deviceName)",
            "global memory     : \(props.major).\(props.minor)",
            "compute capability: \(props.totalGlobalMem / (1024 * 1024)) MB",
            "multiprocessors   : \(props.multiProcessorCount)",
            "memory addressing : \(memoryType)",
        ]
    }
}
