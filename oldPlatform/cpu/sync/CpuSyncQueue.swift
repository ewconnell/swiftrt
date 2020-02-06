//******************************************************************************
// Copyright 2019 Google LLC
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

// @Target(type:"cpu", appliedTo:"CpuSynchronousQueue", protocol: DeviceFunctions)
public final class CpuSynchronousQueue: CpuQueueProtocol, LocalDeviceQueue {
	// protocol properties
    public var trackingId: Int
    public var defaultQueueEventOptions: QueueEventOptions
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    
    /// used to detect accidental queue access by other threads
    @usableFromInline
    let creatorThread: Thread

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(logInfo: LogInfo, device: ComputeDevice, id: Int)
    {
        // create a completion event
        self.logInfo = logInfo
        self.device = device
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        let path = logInfo.namePath
        trackingId = 0
        trackingId = ObjectTracker.global
            .register(self, namePath: path, isStatic: true)
        
        diagnostic("\(createString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: .queueAlloc)
    }
    
    //--------------------------------------------------------------------------
    /// deinit
    /// waits for the queue to finish
    @inlinable
    deinit {
        assert(Thread.current === creatorThread,
               "Queue has been captured and is being released by a " +
            "different thread. Probably by a queued function on the queue.")

        diagnostic("\(releaseString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: [.queueAlloc])
        
        // release
        ObjectTracker.global.remove(trackingId: trackingId)
    }

    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable
    public func createEvent(options: QueueEventOptions) throws -> QueueEvent {
        let event = CpuSyncEvent(options: options, timeout: timeout)
        diagnostic("\(createString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @inlinable
    @discardableResult
    public func record(event: QueueEvent) throws -> QueueEvent {
        guard lastError == nil else { throw lastError! }
        let event = event as! CpuSyncEvent
        diagnostic("\(recordString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            event.recordedTime = Date()
        }
        return event
    }

    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    @inlinable
    public func wait(for event: QueueEvent) throws {
        guard !event.occurred else { return }
        guard lastError == nil else { throw lastError! }
        diagnostic("\(waitString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        try event.wait()
    }

    //--------------------------------------------------------------------------
    // waitUntilQueueIsComplete
    // the synchronous queue completes work as it is queued,
    // so it is always complete
    @inlinable
    public func waitUntilQueueIsComplete() throws { }
    
    //--------------------------------------------------------------------------
    /// perform indexed copy from source view to result view
    @inlinable
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
        // if the queue is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        view.map(into: &result) { $0 }
    }

    //--------------------------------------------------------------------------
    /// copies from one device array to another
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from otherArray: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(array.buffer.count == otherArray.buffer.count,
               "buffer sizes don't match")
        array.buffer.copyMemory(from: UnsafeRawBufferPointer(otherArray.buffer))
    }

    //--------------------------------------------------------------------------
    /// copies a host buffer to a device array
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from hostBuffer: UnsafeRawBufferPointer) throws
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
                          from array: DeviceArray) throws
    {
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        hostBuffer.copyMemory(from: UnsafeRawBufferPointer(array.buffer))
    }

    //--------------------------------------------------------------------------
    /// fills the device array with zeros
    @inlinable
    public func zero(array: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        array.buffer.initializeMemory(as: UInt8.self, repeating: 0)
    }
    
    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the queue by sleeping a duration of
    /// x.count * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
        where T: TensorView
    {
        let delay = TimeInterval(x.count) * timePerElement
        delayQueue(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayQueue(atLeast:
    /// causes the queue to sleep for the specified interval for testing
    public func delayQueue(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }
    
    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        let error = DeviceError.queueError(idPath: [], message: "testError")
        device.report(error)
    }
}
