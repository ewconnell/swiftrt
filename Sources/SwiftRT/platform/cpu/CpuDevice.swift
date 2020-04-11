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
/// CpuDevice
public final class CpuDevice: PlatformDevice {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    public let queues: [DeviceQueue]

    @inlinable
    public init(parent logInfo: LogInfo, memoryType: MemoryType, id: Int)
    {
        self.id = id
        self.name = "cpu:\(id)"
        self.logInfo = logInfo.flat(name)
        self.memoryType = memoryType
        self.queues = [CpuQueue(id: 0,
                                parent: self.logInfo,
                                deviceId: id,
                                deviceName: name,
                                memoryType: memoryType)]
    }
}

//==============================================================================
/// CpuQueue
/// a final version of the default device queue which executes functions
/// synchronously on the cpu
public final class CpuQueue: DeviceQueue { }

