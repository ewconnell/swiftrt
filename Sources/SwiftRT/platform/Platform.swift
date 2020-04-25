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
    associatedtype Device: PlatformDevice

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
/// the type used for memory indexing on discreet devices
public typealias DeviceIndex = Int32

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// StorageOrder
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
/// These names are following the numpy naming convention
public enum StorageOrder: Int, Codable {
    /// dynamic decision based on api
    case A
    /// C style row major memory layout
    case C
    /// Fortran style column major memory layout
    case F
    /// more expressive aliases
    public static let rowMajor = C, colMajor = F
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
/// PlatformDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol PlatformDevice: class, Logger {
    associatedtype Queue: PlatformDeviceQueue
    
    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// name used logging
    var name: String { get }
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }
    /// specifies the type of device memory for data transfer
    var memoryType: MemoryType { get }
}

public protocol PlatformDeviceQueue: class {
    var deviceId: Int { get }
}

//==============================================================================
/// DeviceMemory
public struct DeviceMemory {
    /// base address and size of buffer
    public let buffer: UnsafeMutableRawBufferPointer
    /// function to free the memory
    public let deallocate: () -> Void
    /// specifies the device memory type for data transfer
    public let memoryType: MemoryType
    /// version
    public var version: Int
    
    @inlinable
    public init(buffer: UnsafeMutableRawBufferPointer,
                memoryType: MemoryType,
                _ deallocate: @escaping () -> Void)
    {
        self.buffer = buffer
        self.memoryType = memoryType
        self.version = -1
        self.deallocate = deallocate
    }
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
    
    @inlinable public init() { self.rawValue = 0 }
    @inlinable public init(rawValue: Int) { self.rawValue = rawValue }
}

public enum QueueEventError: Error {
    case timedOut
}

//==============================================================================
/// MemoryType
public enum MemoryType {
    case unified, discreet
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
    case initializeFailed
    case queueError(idPath: [Int], message: String)
    case timeout(idPath: [Int], message: String)
}

//==============================================================================
/// TensorType protocol
/// an n-dimensional collection of elements
/// Currently there is only one tensor type, so these protocols are not
/// needed. They are kept in place for future experimentation.
///
public protocol TensorType: Collection, CustomStringConvertible, Logging
    where Index == ElementIndex<Shape>
{
    /// the ranked short vector type that defines the collection's dimensions
    associatedtype Shape: TensorShape
    /// the type of element in the collection
    associatedtype Element

    //----------------------------------
    /// the number of elements described by `shape`
    var count: Int { get }
    /// `true` if the tensor represents a single constant Element
    var isSingleElement: Bool { get }
    /// the dimensions of the collection
    var shape: Shape { get }
    /// the order in memory to store materialized Elements. Generator
    /// tensor types maintain this property as a template for dense
    /// result tensors.
    var storageOrder: StorageOrder { get }

    //----------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    func makeIndex(at position: Shape) -> Index

    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: the collection slice
    subscript(lower: Shape, upper: Shape) -> Self { get }

    //----------------------------------
    /// `read`
    /// Synchronizes a collection of materialized elements for reading.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `Collection`
    /// enumeration via `indices` or subscripting.
    func read()
    
    /// `read(queue:
    /// Synchronizes a collection of materialized elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    func read(using queue: DeviceQueue)
}

//==============================================================================
/// MutableTensorType
/// an n-dimensional mutable collection of stored elements
public protocol MutableTensorType: TensorType, MutableCollection
{
    /// `true` if the collection can be shared by multiple writers
    /// without performing copy-on-write
    var isShared: Bool { get }
    
    //----------------------------------
    /// `shared`
    /// returns a copy of `self` that does not perform copy-on-write to enable
    /// multi-threaded writes. If the associated storage is not uniquely
    /// referenced, then a copy will be made before returning the sharable
    /// copy. Subscripted views inherit the `isShared` property
    /// - Returns: a sharable copy of `self`
    mutating func shared() -> Self

    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: the collection slice
    subscript(lower: Shape, upper: Shape) -> Self { get set }
    
    //----------------------------------
    /// `readWrite`
    /// Synchronizes a collection of materialized elements for read write.
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `MutableCollection`
    /// enumeration via `indices` or subscripting.
    mutating func readWrite()

    /// `readWrite(queue:`
    /// Synchronizes a mutable collection of materialized elements
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    mutating func readWrite(using queue: DeviceQueue)
}

