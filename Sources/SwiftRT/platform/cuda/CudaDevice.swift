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
/// CudaDevice
public class CudaDevice: ServiceDevice {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    public var queues: [DeviceQueue]

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
        do {
            for i in 0..<numQueues {
                try queues.append(CudaQueue(id: i, parent: logInfo,
                                            deviceId: self.id,
                                            deviceName: name,
                                            useGpu: !isCpuDevice))
            }
        } catch {
            writeLog("\(error)")
        }
    }
    
    public func allocate(byteCount: Int, heapIndex: Int) -> DeviceMemory {
        fatalError()
    }
}
