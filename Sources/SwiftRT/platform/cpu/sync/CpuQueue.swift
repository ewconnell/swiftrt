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
/// CpuQueue
public struct CpuQueue: DeviceQueue, Logging {
    // properties
    public let arrayReplicaKey: Int
    public let defaultQueueEventOptions: QueueEventOptions
    public let deviceId: Int
    public let id: Int
    public let logInfo: LogInfo
    public let deviceName: String
    public let memoryAddressing: MemoryAddressing
    public let name: String

    /// used to detect accidental queue access by other threads
    @usableFromInline
    let creatorThread: Thread

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                replicationKey: Int,
                deviceId: Int, deviceName: String)
    {
        self.arrayReplicaKey = replicationKey
        self.id = id
        self.name = "q:\(id)"
        self.logInfo = logInfo.child(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.memoryAddressing = .unified

        diagnostic("\(createString) DeviceQueue " +
            "\(deviceName)_\(name)", categories: .queueAlloc)
    }

    
    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable
    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        let event = CpuQueueEvent(options: options)
        diagnostic("\(createString) QueueEvent(\(event.id)) on " +
            "\(device.name)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @inlinable
    @discardableResult
    public func record(event: QueueEvent) -> QueueEvent {
        diagnostic("\(recordString) QueueEvent(\(event.id)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
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
    @inlinable
    public func wait(for event: QueueEvent) {
        guard !event.occurred else { return }
        diagnostic("\(waitString) QueueEvent(\(event.id)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
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
    @inlinable
    public func waitUntilQueueIsComplete() { }

    //--------------------------------------------------------------------------
    /// perform indexed copy from source view to result view
    @inlinable
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
        // if the queue is in an error state, no additional work
        // will be queued
        view.map(into: &result) { $0 }
    }

    //--------------------------------------------------------------------------
    /// copies from one device array to another
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from otherArray: DeviceArray) {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(array.buffer.count == otherArray.buffer.count,
               "buffer sizes don't match")
        array.buffer.copyMemory(from: UnsafeRawBufferPointer(otherArray.buffer))
    }

    //--------------------------------------------------------------------------
    /// copies a host buffer to a device array
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from hostBuffer: UnsafeRawBufferPointer)
    {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        array.buffer.copyMemory(from: hostBuffer)
    }
    
    //--------------------------------------------------------------------------
    /// copies a device array to a host buffer
    @inlinable
    public func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                          from array: DeviceArray)
    {
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        hostBuffer.copyMemory(from: UnsafeRawBufferPointer(array.buffer))
    }

    //--------------------------------------------------------------------------
    /// fills the device array with zeros
    @inlinable
    public func zero(array: DeviceArray) {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        array.buffer.initializeMemory(as: UInt8.self, repeating: 0)
    }
}
