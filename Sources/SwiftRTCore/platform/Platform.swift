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
/// Platform
/// a platform represents a collection of installed devices on a
/// compute node, such as (cpu, cuda, tpu, ...)
public protocol Platform: class, Logger {
    // types
    associatedtype Device: ComputeDevice

    /// specifies how to schedule work on the cpu
    static var defaultCpuQueueMode: DeviceQueueMode { get }
    /// a collection of available compute devices
    var devices: [Device] { get }
    /// name used for logging
    var name: String { get }
    /// the current device queue to direct work
    var queueStack: [Device.Queue] { get set }
}

public extension Platform {
    /// the currently active queue that platform functions will use
    /// - Returns: the current device queue
    @inlinable @_transparent
    var currentDevice: Device {
        devices[queueStack.last!.deviceId]
    }

    /// the currently active queue that platform functions will use
    /// - Returns: the current device queue
    @inlinable @_transparent
    var currentQueue: Device.Queue {
        queueStack.last!
    }
}

//==============================================================================
// queue API
@inlinable public func useCpu() {
    Context.local.platform.useCpu()
}

@inlinable public func use(device: Int, queue: Int = 0) {
    Context.local.platform.use(device: device, queue: queue)
}

@inlinable public func using<R>(device: Int, queue: Int = 0,
                                _ body: () -> R) -> R {
    Context.local.platform.using(device: device, queue: queue, body)
}

@inlinable public func using<R>(queue: Int, _ body: () -> R) -> R {
    Context.local.platform.using(queue: queue, body)
}

// Platform extensions
public extension Platform {
    /// changes the current device/queue to use cpu:0
    @inlinable
    func useCpu() {
        queueStack[queueStack.count - 1] = validQueue(0, 0)
    }
    /// selects the specified device queue for output
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    @inlinable
    func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = validQueue(device, queue)
    }
    /// selects the specified device queue for output within the scope of
    /// the body
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    @inlinable
    func using<R>(device: Int, queue: Int = 0, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(validQueue(device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    /// selects the specified queue on the current device for output
    /// within the scope of the body
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    @inlinable
    func using<R>(queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(validQueue(currentQueue.deviceId, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    
    // peforms a mod on the indexes to guarantee they are mapped into bounds
    @inlinable func validQueue(
        _ deviceId: Int,
        _ queueId: Int
    ) -> Device.Queue {
        let device = devices[deviceId % devices.count]
        return device.queues[queueId % device.queues.count]
    }
}

//==============================================================================
/// the type used for memory indexing on discrete devices
public typealias DeviceIndex = Int32

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionOp
public enum ReductionOp: Int, Codable {
    case add
    case mean
    case mul
    case min
    case max
    case amax
    case asum
    case sqrtSumSquares
    case mulNonZeros
    case compare
}

public typealias ReduceOpFinal<R: MutableCollection> = (R.Element) -> R.Element

//==============================================================================
/// ServiceError
/// platform errors
public enum ServiceError : Error {
    case functionFailure(location: String, message: String)
    case rangeError(String)
}

//==============================================================================
public enum EvaluationMode {
    /// operation is used to perform inference
    case inferring
    /// operation is used to perform training
    case training
}

//==============================================================================
/// ComputeDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ComputeDevice: class, Logger {
    associatedtype Queue: DeviceQueue
    
    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// name used logging
    var name: String { get }
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }
    /// specifies the type of device memory for data transfer
    var memoryType: MemoryType { get }
}

//==============================================================================
/// DeviceMemory
public protocol DeviceMemory: class {
    /// base address and size of buffer
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// device where memory is located
    var deviceId: Int { get }
    /// device where memory is located
    var deviceName: String { get }
    /// mutable raw pointer to memory buffer to simplify driver calls
    var pointer: UnsafeMutableRawPointer { get }
    /// specifies the device memory type for data transfer
    var type: MemoryType { get }
    /// version
    var version: Int { get set }
}

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `DeviceQueue`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent: class {
    /// the id of the event for diagnostics
    var id: Int { get }
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    /// the last time the event was recorded
    var recordedTime: Date? { get set }

    /// measure elapsed time since another event
    func elapsedTime(since other: QueueEvent) -> TimeInterval?
    
    /// signals that the event has occurred
    func signal()
    
    /// will block the caller until the timeout has elapsed if one
    /// was specified during init, otherwise it will block forever
    func wait()
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
    
    @inlinable public init() { self.rawValue = 0 }
    @inlinable public init(rawValue: Int) { self.rawValue = rawValue }
}

public enum QueueEventError: Error {
    case timedOut
}

//==============================================================================
/// MemoryType
public enum MemoryType {
    /// the memory is unified with the cpu address space
    case unified
    /// the memory is in a discrete memory address space on another device
    /// and is not directly accessible by the cpu
    case discrete
}

public enum DeviceQueueMode {
    /// the device queue schedule work asynchronously
    case async
    /// the device queue will execute work immediately before returning
    case sync
}

//==============================================================================
/// QueueId
/// a unique service device queue identifier that is used to index
/// through the service device tree for directing workflow
public struct QueueId {
    public let device: Int
    public let queue: Int
    
    @inlinable public init(_ device: Int, _ queue: Int) {
        self.device = device
        self.queue = queue
    }
}

//==============================================================================
// assert messages
public let _messageQueueThreadViolation =
"a queue can only be accessed by the thread that created it"

//==============================================================================
/// DeviceError
public enum DeviceError : Error {
    case allocationFailed
    case initializeFailed
    case queueError(idPath: [Int], message: String)
    case timeout(idPath: [Int], message: String)
}

//==============================================================================
/// ScalarType
/// Used primarily for serialization, C APIs, and Cuda kernels
public enum ScalarType: Int {
    // integers
    case real8U, real8I, real16U, real16I, real32U, real32I, real64U, real64I
    // floats
    case real16F, real32F, real64F
    // non numeric
    case bool
}

public protocol ScalarElement: StorageElement {
    static var type: ScalarType { get }
    static var zeroPointer: UnsafeRawPointer { get }
    static var onePointer: UnsafeRawPointer { get }
}

extension Int8: ScalarElement {
    @inlinable public static var type: ScalarType { .real8I }
    
    public static var zero: Self = 0
    @inlinable public
    static var zeroPointer: UnsafeRawPointer { UnsafeRawPointer(&zero) }
    
    public static var one: Self = 1
    @inlinable public
    static var onePointer: UnsafeRawPointer { UnsafeRawPointer(&one) }
}

extension UInt8: ScalarElement {
    @inlinable public static var type: ScalarType { .real8U }
    
    public static var zero: Self = 0
    @inlinable public
    static var zeroPointer: UnsafeRawPointer { UnsafeRawPointer(&zero) }
    
    public static var one: Self = 1
    @inlinable public
    static var onePointer: UnsafeRawPointer { UnsafeRawPointer(&one) }
}

extension Float: ScalarElement {
    @inlinable public static var type: ScalarType { .real32F }
    
    public static var zero: Self = 0
    @inlinable public
    static var zeroPointer: UnsafeRawPointer { UnsafeRawPointer(&zero) }
    
    public static var one: Self = 1
    @inlinable public
    static var onePointer: UnsafeRawPointer { UnsafeRawPointer(&one) }
}

extension Double: ScalarElement {
    @inlinable public static var type: ScalarType { .real64F }
    
    public static var zero: Self = 0
    @inlinable public
    static var zeroPointer: UnsafeRawPointer { UnsafeRawPointer(&zero) }
    
    public static var one: Self = 1
    @inlinable public
    static var onePointer: UnsafeRawPointer { UnsafeRawPointer(&one) }
}

