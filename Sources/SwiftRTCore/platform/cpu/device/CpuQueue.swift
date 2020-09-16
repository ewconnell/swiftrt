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
/// CpuFunctions
public protocol CpuFunctions { }

//==============================================================================
/// CpuQueue
/// a final version of the default device queue which executes functions
/// synchronously on the cpu
public final class CpuQueue: DeviceQueue, CpuFunctions
{
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceIndex: Int
    public let id: Int
    public let memoryType: MemoryType
    public let mode: DeviceQueueMode
    public let name: String
    public let group: DispatchGroup
    public let queue: DispatchQueue
    public let queueLimit: DispatchSemaphore
    public let usesCpu: Bool

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(
        deviceIndex: Int,
        name: String,
        queueMode: DeviceQueueMode,
        memoryType: MemoryType
    ) {
        self.deviceIndex = deviceIndex
        self.name = name
        self.memoryType = memoryType
        id = Context.nextQueueId
        creatorThread = Thread.current
        defaultQueueEventOptions = QueueEventOptions()
        mode = queueMode
        group = DispatchGroup()
        queue = DispatchQueue(label: "\(name)", attributes: .concurrent)
        queueLimit = DispatchSemaphore(value: ProcessInfo().activeProcessorCount)
        usesCpu = true
    }
    
    deinit {
        // make sure all scheduled work is complete before exiting
        waitForCompletion()
        diagnostic(.release, "queue: \(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------
    // allocate
    @inlinable public func allocate(
        byteCount: Int,
        heapIndex: Int = 0
    ) -> DeviceMemory {
        // allocate a host memory buffer suitably aligned for any type
        let buffer = UnsafeMutableRawBufferPointer
                .allocate(byteCount: byteCount,
                          alignment: MemoryLayout<Int>.alignment)
        return CpuDeviceMemory(deviceIndex, buffer, memoryType)
    }
}
