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
        let processorCount = ProcessInfo.processInfo.activeProcessorCount
        self.deviceIndex = deviceIndex
        self.name = name
        self.memoryType = memoryType
        id = Context.nextQueueId
        creatorThread = Thread.current
        defaultQueueEventOptions = QueueEventOptions()
        mode = queueMode
        group = DispatchGroup()
        queue = DispatchQueue(label: name)
        queueLimit = DispatchSemaphore(value: processorCount)
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

    //--------------------------------------------------------------------------
    @inlinable public func recordEvent() -> CpuEvent {
        let event = CpuEvent(mode == .sync ? .signaled : .blocking) 
        if mode == .async {
            queue.async(group: group) {
                event.signal()
            }
        }
        return event
    }
    
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// causes this queue to wait until the event has occurred
    @inlinable public func wait(for event: CpuEvent) {
        diagnostic(.wait, "\(name) will wait for event(\(event.id))",
                   categories: .queueSync)
        if mode == .async {
            queue.async(group: group) {
                event.wait()
            }
        } else {
            event.wait()
        }
    }
    
    //--------------------------------------------------------------------------
    // waitForCompletion
    // the synchronous queue completes work as it is queued,
    // so it is always complete
    @inlinable public func waitForCompletion() {
        if mode == .async {
            group.wait()
        }
    }
}
