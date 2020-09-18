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
/// DeviceQueue
/// Used to schedule and execute computations
public protocol DeviceQueue: Logging {
    /// the thread that created this queue. Used to protect against
    /// accidental cross thread access
    var creatorThread: Thread { get }
    /// used to configure event creation
    var defaultQueueEventOptions: QueueEventOptions { get set }
    /// the index of the parent device in the `devices` collection
    var deviceIndex: Int { get }
    /// the name of the associated device, used in diagnostics
    var deviceName: String { get }
    /// the async cpu `queue` dispatch group used to wait for queue completion
    var group: DispatchGroup { get }
    /// a unique queue id used to identify data movement across queues
    var id: Int { get }
    /// the type of memory associated with the queue's device
    var memoryType: MemoryType { get }
    /// specifies if work is queued sync or async
    var mode: DeviceQueueMode { get }
    /// the name of the queue for diagnostics
    var name: String { get }
    /// the asynchronous operation queue
    var queue: DispatchQueue { get }
    /// `true` if the queue executes work on the cpu
    var usesCpu: Bool { get }

    //--------------------------------------------------------------------------
    /// allocate(alignment:byteCount:heapIndex:
    /// allocates a block of memory on the associated device. If there is
    /// insufficient storage, a `DeviceError` will be thrown.
    /// - Parameters:
    ///  - byteCount: the number of bytes to allocate
    ///  - heapIndex: reserved for future use. Should be 0 for now.
    /// - Returns: a device memory object
    func allocate(byteCount: Int, heapIndex: Int) -> DeviceMemory
    
    //--------------------------------------------------------------------------
    /// copyAsync(src:dst:
    /// copies device memory and performs marshalling if needed
    /// - Parameters:
    ///  - src: the source buffer
    ///  - dst: the destination buffer
    func copyAsync(from src: DeviceMemory, to dst: DeviceMemory)
    
    // /// createEvent(options:
    // /// creates a queue event used for synchronization and timing measurements
    // /// - Parameters:
    // ///  - options: event creation options
    // /// - Returns: a new queue event
    // func createEvent(options: QueueEventOptions) -> PlatformType.Event
    
    /// record(event:
    /// adds `event` to the queue and returns immediately
    /// - Parameters:
    ///  - event: the event to record
    /// - Returns: `event` so that calls to `record` can be nested
    func record(event: PlatformType.Event) -> PlatformType.Event
    
    /// wait(event:
    /// queues an operation to wait for the specified event. This function
    /// does not block the calller if queue `mode` is `.async`
    /// - Parameters:
    ///  - event: the event to wait for
    func wait(for event: PlatformType.Event)
    
    /// waitForCompletion
    /// blocks the caller until all events in the queue have completed
    func waitForCompletion()
}

//==============================================================================
// default implementation
extension DeviceQueue {
    //--------------------------------------------------------------------------
    /// allocate(byteCount:
    @inlinable public func allocate(_ byteCount: Int) -> DeviceMemory {
        allocate(byteCount: byteCount, heapIndex: 0)
    }

    //--------------------------------------------------------------------------
    /// copy
    @inlinable public func copyAsync(
        from src: DeviceMemory, 
        to dst: DeviceMemory
    ) {
        cpu_copyAsync(from: src, to: dst)
    }

    @inlinable public func cpu_copyAsync(
        from src: DeviceMemory, 
        to dst: DeviceMemory
    ) {
        if mode == .async {
            queue.async(group: group) {
                dst.buffer.copyMemory(from: UnsafeRawBufferPointer(src.buffer))
            }
        } else {
            dst.buffer.copyMemory(from: UnsafeRawBufferPointer(src.buffer))
        }
    }

    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable public func createEvent(
        options: QueueEventOptions
    ) -> PlatformType.Event {
        PlatformType.Event()
    }

    @inlinable public func createEvent() -> PlatformType.Event {
        createEvent(options: defaultQueueEventOptions)
    }
    
    //--------------------------------------------------------------------------
    /// deviceName
    /// returns a diagnostic name for the device assoicated with this queue
    @inlinable public var deviceName: String {
        Context.local.platform.devices[deviceIndex].name
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    @inlinable public func record(event: PlatformType.Event) -> PlatformType.Event {
        diagnostic(.record, "event(\(event.id)) on \(name)",
                   categories: .queueSync)
        
        // record the event
        if mode == .sync {
            event.signal()
        } else {
            queue.async(group: group) {
                event.signal()
            }
        }
        return event
    }
    
    @discardableResult
    @inlinable public func record() -> PlatformType.Event {
        record(event: PlatformType.Event())
    }

    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    @inlinable public func wait(for event: PlatformType.Event) {
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
