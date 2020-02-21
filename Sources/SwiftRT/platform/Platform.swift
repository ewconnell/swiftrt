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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Platform
/// Manages the scope for the current devices, log, and error handlers
public class Platform {
    /// the log output object
    @usableFromInline static var logWriter: Log = Log()
    /// the current compute platform for the thread
    //    @usableFromInline var platform: PlatformService
    /// a platform instance unique id for queue events
    @usableFromInline static var queueEventCounter: Int = 0
    
    
    // maybe thread local
    public static var service: PlatformAPI = CpuService()
    
    //--------------------------------------------------------------------------
    /// the Platform log writing object
    @inlinable public static var log: Log {
        get { logWriter }
        set { logWriter = newValue }
    }
    /// a counter used to uniquely identify queue events for diagnostics
    @inlinable static var nextQueueEventId: Int {
        queueEventCounter += 1
        return queueEventCounter
    }
    
    //--------------------------------------------------------------------------
    /// returns the thread local instance of the queues stack
    @usableFromInline
    static var threadLocal: Platform {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = Platform()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }
    
    //--------------------------------------------------------------------------
    /// thread data key
    @usableFromInline
    static let key: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key) {
            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
            #else
            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
            #endif
        }
        return key
    }()
}

//==============================================================================
/// PlatformService
/// a compute service represents a category of installed devices on the
/// platform, such as (cpu, cuda, tpu, ...)
public protocol PlatformService: PlatformAPI, Logger {
    // types
    associatedtype Device: ServiceDevice
    associatedtype MemoryManager: MemoryManagement

    /// a collection of available compute devices
    var devices: [Device] { get }
    /// service memory manager
    var memory: MemoryManager { get }
    /// name used for logging
    var name: String { get }
    /// the current device and queue to direct work
    var queueStack: [QueueId] { get set }
}

public extension PlatformService {
    /// the currently active queue that platform service functions will use
    /// - Returns: the current device queue
    @inlinable
    var currentQueue: DeviceQueue {
        let queueId = queueStack.last!
        return devices[queueId.device].queues[queueId.queue]
    }
    
    //--------------------------------------------------------------------------
    /// read(tensor:
    /// gains synchronized read only access to the tensor `elementBuffer`
    /// - Parameter tensor: the tensor to read
    /// - Returns: an `ElementBuffer` that can be used to iterate the shape
    @inlinable
    func read<T>(_ tensor: T) -> ElementBuffer<T.Element, T.Shape>
        where T: TensorView
    {
        let buffer = memory.read(tensor.elementBuffer, of: T.Element.self,
                                 at: tensor.offset,
                                 count: tensor.shape.spanCount,
                                 using: currentQueue)
        return ElementBuffer(tensor.shape, buffer)
    }

    //--------------------------------------------------------------------------
    /// write(tensor:willOverwrite:
    /// gains synchronized read write access to the tensor `elementBuffer`
    /// - Parameter tensor: the tensor to read
    /// - Parameter willOverwrite: `true` if all elements will be written
    /// - Returns: an `ElementBuffer` that can be used to iterate the shape
    @inlinable
    func write<T>(_ tensor: T, willOverwrite: Bool = true)
        -> MutableElementBuffer<T.Element, T.Shape> where T: TensorView
    {
        let buffer = memory.readWrite(tensor.elementBuffer,
                                      of: T.Element.self,
                                      at: tensor.offset,
                                      count: tensor.shape.spanCount,
                                      willOverwrite: willOverwrite,
                                      using: currentQueue)
        return MutableElementBuffer(tensor.shape, buffer)
    }

    //--------------------------------------------------------------------------
    /// `createResult(shape:name:`
    /// creates a new tensor like the one specified and access to it's
    /// `elementBuffer`
    /// - Parameter other: a tensor to use as a template
    /// - Parameter shape: the shape of the tensor to create
    /// - Parameter name: an optional name for the new tensor
    /// - Returns: a tensor and an associated `MutableElementBuffer`
    /// that can be used to iterate the shape
    @inlinable
    func createResult<T>(like other: T, with shape: T.Shape, name: String? = nil)
        -> (T, MutableElementBuffer<T.Element, T.Shape>) where T: TensorView
    {
        let result = other.createDense(with: shape.dense, name: name)
        return (result, write(result))
    }

    //--------------------------------------------------------------------------
    /// `createResult(other:name:`
    /// creates a new tensor like the one specified and access to it's
    /// `elementBuffer`
    /// - Parameter other: a tensor to use as a template
    /// - Parameter name: an optional name for the new tensor
    /// - Returns: a tensor and an associated `MutableElementBuffer`
    /// that can be used to iterate the shape
    @inlinable
    func createResult<T>(like other: T, name: String? = nil)
        -> (T, MutableElementBuffer<T.Element, T.Shape>) where T: TensorView
    {
        createResult(like: other, with: other.shape, name: name)
    }
}

//==============================================================================
// queue API
@inlinable public func useCpu() {
    Platform.service.useCpu()
}

@inlinable public func use(device: Int, queue: Int = 0) {
    Platform.service.use(device: device, queue: queue)
}

@inlinable public func using<R>(device: Int, queue: Int = 0,
                                _ body: () -> R) -> R {
    Platform.service.using(device: device, queue: queue, body)
}

@inlinable public func using<R>(queue: Int, _ body: () -> R) -> R {
    Platform.service.using(queue: queue, body)
}

// Platform extensions
extension PlatformService {
    /// changes the current device/queue to use cpu:0
    @inlinable
    public mutating func useCpu() {
        queueStack[queueStack.count - 1] = QueueId(0, 0)
    }
    /// selects the specified device queue for output
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    @inlinable
    public mutating func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = ensureValidId(device, queue)
    }
    /// selects the specified device queue for output within the scope of
    /// the body
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    @inlinable
    public mutating func using<R>(device: Int,
                                  queue: Int = 0, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(ensureValidId(device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    /// selects the specified queue on the current device for output
    /// within the scope of the body
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    @inlinable
    public mutating func using<R>(queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        let current = queueStack.last!
        queueStack.append(ensureValidId(current.device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    // peforms a mod on the indexes to guarantee they are mapped into bounds
    @inlinable
    public func ensureValidId(_ deviceId: Int, _ queueId: Int) -> QueueId {
        let device = deviceId % devices.count
        let queue = queueId % devices[device].queues.count
        return QueueId(device, queue)
    }
}

//==============================================================================
/// ServiceDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ServiceDevice: Logger {
    associatedtype Queue: DeviceQueue

    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// name used logging
    var name: String { get }
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }

    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    /// - Parameter byteCount: the number of bytes to allocate on the device
    /// - Parameter heapIndex: the index of the heap to use
    /// - Parameter zero: `true` to inialize the array to zero
    func createArray(byteCount: Int, heapIndex: Int, zero: Bool) -> DeviceArray
    /// creates a device array from a uma buffer.
    /// - Parameter buffer: a read only byte buffer in the device's
    /// address space
    func createReferenceArray(buffer: UnsafeRawBufferPointer) -> DeviceArray
    /// creates a device array from a uma buffer.
    /// - Parameter buffer: a read write byte buffer in the device's
    /// address space
    func createMutableReferenceArray(buffer: UnsafeMutableRawBufferPointer)
        -> DeviceArray
}

//==============================================================================
/// DeviceQueue
/// A device queue is an asynchronous sequential list of commands to be
/// executed on the associated device.
public protocol DeviceQueue: Logger, DeviceFunctions {
    /// the thread that created this queue. Used to detect accidental access
    var creatorThread: Thread { get }
    /// options to use when creating queue events
    var defaultQueueEventOptions: QueueEventOptions { get }
    /// the device id that this queue is associated with
    var deviceId: Int { get }
    /// the id of the device for example queue:0, queue:1, ...
    var id: Int { get }
    /// name used logging
    var deviceName: String { get }
    /// specifies the type of associated device memory
    var memoryAddressing: MemoryAddressing { get }
    /// name used logging
    var name: String { get }

    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a QueueEvent
    func createEvent(options: QueueEventOptions) -> QueueEvent
    /// queues a queue event op. When executed the event is signaled
    @discardableResult
    func record(event: QueueEvent) -> QueueEvent
    /// records an op on the queue that will perform a queue blocking wait
    /// when it is processed
    func wait(for event: QueueEvent)
    /// blocks the calling thread until the queue queue has completed all work
    func waitUntilQueueIsComplete()
}

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `DeviceQueue`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent {
    /// the id of the event for diagnostics
    var id: Int { get }
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    /// the last time the event was recorded
    var recordedTime: Date? { get set }

    /// measure elapsed time since another event
    func elapsedTime(since other: QueueEvent) -> TimeInterval?
    /// will block the caller until the timeout has elapsed if one
    /// was specified during init, otherwise it will block forever
    func wait() throws
}

//==============================================================================
public extension QueueEvent {
    /// elapsedTime
    /// computes the timeinterval between two queue event recorded times
    /// - Parameter other: the other event used to compute the interval
    /// - Returns: the elapsed interval. Will return `nil` if this event or
    ///   the other have not been recorded.
    @inlinable
    func elapsedTime(since other: QueueEvent) -> TimeInterval? {
        guard let time = recordedTime,
            let other = other.recordedTime else { return nil }
        return time.timeIntervalSince(other)
    }
}

//==============================================================================
/// QueueEventOptions
public struct QueueEventOptions: OptionSet {
    public let rawValue: Int
    public static let timing       = QueueEventOptions(rawValue: 1 << 0)
    public static let interprocess = QueueEventOptions(rawValue: 1 << 1)
    
    @inlinable
    public init() { self.rawValue = 0 }
    
    @inlinable
    public init(rawValue: Int) { self.rawValue = rawValue }
}

public enum QueueEventError: Error {
    case timedOut
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking {
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the device id that this array is associated with
    var deviceId: Int { get }
    /// name used logging
    var deviceName: String { get }
    /// `true` if the array is read only
    var isReadOnly: Bool { get }
    /// specifies the type of associated device memory
    var memoryAddressing: MemoryAddressing { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }
}

//==============================================================================
/// MemoryAddressing
public enum MemoryAddressing {
    case unified, discreet
}

//==============================================================================
/// QueueId
/// a unique service device queue identifier that is used to index
/// through the service device tree for directing workflow
public struct QueueId {
    public let device: Int
    public let queue: Int
    public init(_ device: Int, _ queue: Int) {
        self.device = device
        self.queue = queue
    }
}

//==============================================================================
// assert messages
let _messageQueueThreadViolation =
"a queue can only be accessed by the thread that created it"

