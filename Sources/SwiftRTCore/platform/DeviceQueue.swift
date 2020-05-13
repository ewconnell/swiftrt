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
    /// the id of the associated device
    var deviceId: Int { get }
    /// the name of the associated device, used in diagnostics
    var deviceName: String { get }
    /// the id of the queue
    var id: Int { get }
    /// the logging configuration for the queue
    var logInfo: LogInfo { get }
    /// the type of memory associated with the queue's device
    var memoryType: MemoryType { get }
    /// specifies if work is queued sync or async
    var mode: DeviceQueueMode { get }
    /// the name of the queue for diagnostics
    var name: String { get }

    //--------------------------------------------------------------------------
    /// allocate(type:count:heapIndex:
    /// allocates a block of memory on the associated device. If there is
    /// insufficient storage, a `DeviceError` will be thrown.
    /// - Parameters:
    ///  - type: the type of element that will be stored. This ensures
    ///    correct storage byte size and alignment
    ///  - count: the number of elements to store
    ///  - heapIndex: reserved for future use. Should be 0 for now.
    /// - Returns: a device memory object
    func allocate<Element>(
        _ type: Element.Type,
        count: Int,
        heapIndex: Int
    ) throws -> DeviceMemory<Element>
    
    /// createEvent(options:
    /// creates a queue event used for synchronization and timing measurements
    /// - Parameters:
    ///  - options: event creation options
    /// - Returns: a new queue event
    func createEvent(options: QueueEventOptions) -> QueueEvent
    
    /// record(event:
    /// adds `event` to the queue and returns immediately
    /// - Parameters:
    ///  - event: the event to record
    /// - Returns: `event` so that calls to `record` can be nested
    func record(event: QueueEvent) -> QueueEvent
    
    /// wait(event:
    /// queues an operation to wait for the specified event. This function
    /// does not block the calller if queue `mode` is `.async`
    /// - Parameters:
    ///  - event: the event to wait for
    func wait(for event: QueueEvent)
    
    /// waitUntilQueueIsComplete
    /// blocks the caller until all events in the queue have completed
    func waitUntilQueueIsComplete()
}

//==============================================================================
// default implementation
extension DeviceQueue {
    //--------------------------------------------------------------------------
    // allocate
    @inlinable public func allocate<Element>(
        _ type: Element.Type,
        count: Int,
        heapIndex: Int = 0
    ) throws -> DeviceMemory<Element> {
        // allocate an aligned host memory buffer
        let buff = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        return DeviceMemory(buff, memoryType, { buff.deallocate() })
    }
    
    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable public func createEvent(
        options: QueueEventOptions
    ) -> QueueEvent {
        let event = CpuQueueEvent(options: options)
        diagnostic("\(createString) QueueEvent(\(event.id)) on " +
                    "\(deviceName)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// deviceName
    /// returns a diagnostic name for the device assoicated with this queue
    @inlinable public var deviceName: String {
        Context.local.platform.devices[deviceId].name
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    @inlinable public func record(
        event: QueueEvent
    ) -> QueueEvent {
        diagnostic("\(recordString) QueueEvent(\(event.id)) on " +
                    "\(deviceName)_\(name)", categories: .queueSync)
        
        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            var timeStampedEvent = event
            timeStampedEvent.recordedTime = Date()
            return timeStampedEvent
        } else {
            return event
        }
    }
    
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    @inlinable public func wait(
        for event: QueueEvent
    ) {
        guard !event.occurred else { return }
        diagnostic("\(waitString) QueueEvent(\(event.id)) on " +
                    "\(deviceName)_\(name)", categories: .queueSync)
        do {
            try event.wait()
        } catch {
            // there is no recovery here
            writeLog("\(error)")
            fatalError()
        }
    }
    
    //--------------------------------------------------------------------------
    // waitUntilQueueIsComplete
    // the synchronous queue completes work as it is queued,
    // so it is always complete
    @inlinable public func waitUntilQueueIsComplete() { }
}
