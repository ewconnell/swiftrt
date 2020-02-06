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
// Platform Abstraction Protocols
//
//  ComputePlatform
//    ComputeService
//      cpu and accelerators[]
//        devices[]
//          ComputeDevice (dev:0, dev:1, ...)
//            DeviceArray
//            DeviceQueue
//              QueueEvent
//
import Foundation

//==============================================================================
/// ComputePlatform
/// The root collection of compute resources available to the application
/// on a given machine
public protocol ComputePlatform: Logger {
    // types
    associatedtype Service: ComputeService
    
    //--------------------------------------------------------------------------
    // properties
    /// the platform id. Usually zero, but can be assigned in case a higher
    /// level object (e.g. cluster) will maintain a platform collection
    var id: Int { get }
    /// name used logging
    var name: String { get }
    /// the local device compute service
    var service: Service { get }
    /// the current device and queue to direct work
    var queueStack: [(device: Int, queue: Int)] { get set }
}

//==============================================================================
/// ComputePlatform extensions for queue stack manipulation
public extension ComputePlatform {
    /// the currently active queue that API functions will use
    /// - Returns: the current device queue
    var currentQueue: Service.Device.Queue {
        let (device, queue) = queueStack.last!
        return service.devices[device].queues[queue]
    }
    /// changes the current device/queue to use cpu:0
    mutating func useCpu() {
        queueStack[queueStack.count - 1] = (0, 0)
    }
    /// selects the specified device queue for output
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    mutating func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = makeValidQueueIndexes(device, queue)
    }
    /// selects the specified device queue for output within the scope of
    /// the body
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    mutating func using<R>(device: Int, queue: Int = 0, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(makeValidQueueIndexes(device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    /// selects the specified queue on the current device for output
    /// within the scope of the body
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    mutating func using<R>(queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(makeValidQueueIndexes(queueStack.last!.device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    // peforms a mod on the indexes to guarantee they are mapped into bounds
    func makeValidQueueIndexes(_ device: Int, _ queue: Int) -> (Int, Int){
        let deviceIndex = device % service.devices.count
        let queueIndex = queue % service.devices[deviceIndex].queues.count
        return (deviceIndex, queueIndex)
    }
}

//==============================================================================
/// ComputeService
/// a compute service represents a category of installed devices on the
/// platform, such as (cpu, cuda, tpu, ...)
public protocol ComputeService: Logger {
    // types
    associatedtype Device: ComputeDevice

    //--------------------------------------------------------------------------
    // properties
    /// a collection of available compute devices
    var devices: [Device] { get }
    /// service id used for logging, usually zero
    var id: Int { get }
    /// name used logging
    var name: String { get }

    //--------------------------------------------------------------------------
    // initializers
    init(parent parentLogInfo: LogInfo, id: Int)
}

//==============================================================================
/// ServiceOptimizer
public protocol ServiceOptimizer: ComputeService {
    associatedtype Service: ComputeService
    
    var service: Service { get }
}

//==============================================================================
/// ComputeDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ComputeDevice: Logger {
    associatedtype Queue: DeviceQueue & DeviceFunctions
    // properties
    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// name used logging
    var name: String { get }
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }
}

//==============================================================================
/// DeviceQueue
/// A device queue is an asynchronous sequential list of commands to be
/// executed on the associated device.
public protocol DeviceQueue: Logger {
    // properties
    /// the id of the device for example queue:0, queue:1, ...
    var id: Int { get }
    /// name used logging
    var deviceName: String { get }
    /// name used logging
    var name: String { get }

    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a QueueEvent
    @inlinable
    func createEvent(options: QueueEventOptions) -> QueueEvent
    /// queues a queue event op. When executed the event is signaled
    @inlinable
    @discardableResult
    func record(event: QueueEvent) -> QueueEvent
    /// records an op on the queue that will perform a queue blocking wait
    /// when it is processed
    @inlinable
    func wait(for event: QueueEvent)
    /// blocks the calling thread until the queue queue has completed all work
    @inlinable
    func waitUntilQueueIsComplete()
}

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `DeviceQueue`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent {
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

//------------------------------------------------------------------------------
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

public extension QueueEvent {
    //--------------------------------------------------------------------------
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
