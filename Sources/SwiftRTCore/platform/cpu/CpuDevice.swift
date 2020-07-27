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
    public let index: Int
    public let memoryType: MemoryType
    public let name: String
    public var queues: [CpuQueue]

    @inlinable public init(index: Int, memoryType: MemoryType) {
        self.index = index
        self.name = "dev:\(index)"
        self.memoryType = memoryType
        self.queues = []
        diagnostic("\(deviceString) create \(name)  memory: \(memoryType)",
                   categories: .device)
        for i in 0..<Context.cpuQueueCount {
            let queue = CpuQueue(
                deviceIndex: index,
                name: "\(name)_q\(i)",
                queueMode: .async,
                memoryType: memoryType)
            queues.append(queue)
        }
    }
}

//==============================================================================
/// CpuDeviceMemory
public final class CpuDeviceMemory: DeviceMemory {
    /// base address and size of buffer
    public let buffer: UnsafeMutableRawBufferPointer
    /// index of device where memory is located
    public let deviceIndex: Int
    /// diagnostic name
    public var name: String?
    /// diagnostic message
    public var releaseMessage: String?
    /// specifies the device memory type for data transfer
    public let type: MemoryType
    /// version
    public var version: Int
    /// `true` if the buffer is a reference
    public let isReference: Bool

    /// mutable raw pointer to memory buffer to simplify driver calls
    @inlinable public var mutablePointer: UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(buffer.baseAddress!)
    }

    /// raw pointer to memory buffer to simplify driver calls
    @inlinable public var pointer: UnsafeRawPointer {
        UnsafeRawPointer(buffer.baseAddress!)
    }

    /// device where memory is located
    @inlinable public var device: PlatformType.Device {
        Context.devices[deviceIndex]
    }

    @inlinable public init(
        _ deviceIndex: Int,
        _ buffer: UnsafeMutableRawBufferPointer,
        _ type: MemoryType,
        isReference: Bool = false
    ) {
        self.deviceIndex = deviceIndex
        self.buffer = buffer
        self.type = type
        self.isReference = isReference
        self.version = -1
        self.name = nil
        self.releaseMessage = nil
    }
    
    @inlinable deinit {
        if !isReference {
            buffer.deallocate()
            #if DEBUG
            if let msg = releaseMessage {
                diagnostic("\(releaseString) \(msg)", categories: .dataAlloc)
            }
            #endif
        }
    }
}
