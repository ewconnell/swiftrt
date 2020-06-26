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
public final class CpuDevice: ComputeDevice {
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    public var queues: [CpuQueue]

    @inlinable public init(
        id: Int,
        parent logInfo: LogInfo,
        memoryType: MemoryType
    ) {
        self.id = id
        self.name = "cpu:\(id)"
        self.logInfo = logInfo.flat(name)
        self.memoryType = memoryType
        self.queues = []
        for _ in 0..<Context.queuesPerDevice {
            queues.append(CpuQueue(parent: self.logInfo,
                                   deviceId: id,
                                   deviceName: name,
                                   memoryType: memoryType,
                                   mode: .async))
        }
    }
}

//==============================================================================
/// CpuDeviceMemory
public final class CpuDeviceMemory: DeviceMemory {
    /// base address and size of buffer
    public let buffer: UnsafeMutableRawBufferPointer
    /// device where memory is located
    public let deviceId: Int
    /// the name of the device for diagnostics
    public let deviceName: String
    /// diagnostic message
    public var releaseMessage: String?
    /// specifies the device memory type for data transfer
    public let type: MemoryType
    /// version
    public var version: Int
    /// `true` if the buffer is a reference
    public let isReference: Bool
    /// mutable raw pointer to memory buffer to simplify driver calls
    @inlinable public var pointer: UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(buffer.baseAddress!)
    }

    /// device where memory is located
    @inlinable public var device: PlatformType.Device {
        Context.devices[deviceId]
    }

    @inlinable public init(
        _ deviceId: Int,
        _ deviceName: String,
        buffer: UnsafeMutableRawBufferPointer,
        isReference: Bool = false
    ) {
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.buffer = buffer
        self.type = .unified
        self.isReference = isReference
        self.version = 0
        self.releaseMessage = nil
    }
    
    @inlinable deinit {
        if !isReference {
            buffer.deallocate()
            if let msg = releaseMessage {
                diagnostic(msg, categories: .dataAlloc)
            }
        }
    }
}
