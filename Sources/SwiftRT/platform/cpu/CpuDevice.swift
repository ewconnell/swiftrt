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
/// CpuQueueProtocol
public protocol CpuQueueProtocol: DeviceQueue {
    init(id: Int,
         parent logInfo: LogInfo,
         addressing: MemoryAddressing,
         deviceId: Int, deviceName: String)
}

//==============================================================================
/// CpuDevice
public struct CpuDevice<Queue>: ServiceDevice
    where Queue: CpuQueueProtocol
{
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let name: String
    public let queues: [Queue]
    
    @inlinable
    public init(parent logInfo: LogInfo, addressing: MemoryAddressing, id: Int)
    {
        let deviceName = "cpu:\(id)"
        self.id = id
        self.name = deviceName
        self.logInfo = logInfo.child(name)
        
        let queues = [Queue(id: 0, parent: self.logInfo,
                            addressing: addressing,
                            deviceId: id, deviceName: name)]
        self.queues = queues
    }

    //--------------------------------------
    // allocate
    public func allocate(byteCount: Int, heapIndex: Int) -> DeviceMemory {
        // allocate a host memory buffer
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<Double>.alignment)

        return DeviceMemory(buffer.baseAddress,
                            byteCount: byteCount,
                            version: 0,
                            memoryAddressing: .unified,
                            { buffer.deallocate() })
    }
}

