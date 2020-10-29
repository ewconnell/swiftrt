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
// Platform types
#if canImport(SwiftRTCuda)
public typealias Platform = CudaPlatform
#else
public typealias Platform = CpuPlatform
#endif

// global ambient properties used for expression evaluation
public var log: Log = Log()
@inlinable public var platform: Platform { Platform.local }
@inlinable public var currentDevice: Platform.Device { platform.currentDevice }
@inlinable public var currentQueue: Platform.Device.Queue { platform.currentQueue }

//==============================================================================
/// ComputePlatform
/// a platform represents a collection of installed devices on a
/// compute node, such as (cpu, cuda, tpu, ...)
public protocol ComputePlatform: class, Logging {
    // types
    associatedtype Device: ComputeDevice
    associatedtype Storage: StorageBuffer

    //--------------------------------------------------------------------------
    // shared state
    /// returns an id to a discrete memory device to support unit tests
    static var discreteMemoryDeviceId: Int { get }
    /// identifies the main thread
    static var mainThread: pthread_t { get }
    /// the time that the platform was first accessed
    static var startTime: Date { get }
    /// queue used to synchronize data interchange with the application thread
    static var syncQueue: Device.Queue { get }

    //--------------------------------------------------------------------------
    // instance state
    /// a collection of available compute devices
    var devices: [Device] { get }
    /// name used for logging
    var name: String { get }
    /// the current device queue to direct work
    var queueStack: [Device.Queue] { get set }

    //--------------------------------------------------------------------------
    /// used by random number generators
    static var lastRandomSeed: RandomSeed { get set }
    /// counter for unique buffer ids
    static var objectId: AtomicCounter { get }
    /// a platform instance unique id for queue events
    static var queueId: AtomicCounter { get }
    /// a platform instance unique id for queue events
    static var eventId: AtomicCounter { get }
}

public extension ComputePlatform {
    /// `true` if the caller is the main thread
    @inlinable var isMainThread: Bool { pthread_self() == Self.mainThread }
    
    /// the currently active queue for expression evaluation
    @inlinable var currentQueue: Device.Queue {
        isMainThread ? queueStack.last! : Self.syncQueue
    }

    /// the currently active device
    @inlinable var currentDevice: Device {
        isMainThread ? devices[queueStack.last!.deviceIndex] : devices[0]
    }
    
    /// selects the specified device queue for output
    /// - Parameters:
    ///  - device: the device to use. Device 0 is the cpu
    ///  - queue: the queue on the device to use
    @inlinable func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = validQueue(device, queue)
    }
    
    /// selects the application thread data interchange queue within
    /// the scope of the body
    @inlinable func useSyncQueue() {
        queueStack[queueStack.count - 1] = Self.syncQueue
    }
    
    /// selects the application thread data interchange queue within
    /// the scope of the body
    /// - Parameters:
    ///  - body: a closure where the device queue will be used
    @inlinable func usingSyncQueue<R>(_ body: () -> R) -> R {
        queueStack.append(Self.syncQueue)
        defer { _ = queueStack.popLast() }
        return body()
    }

    /// selects the specified device queue for output within the scope of
    /// the body
    /// - Parameters:
    ///  - device: the device to use. Device 0 is the cpu
    ///  - queue: the queue on the device to use
    ///  - body: a closure where the device queue will be used
    @inlinable func using<R>(device: Int, queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(validQueue(device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    
    /// selects the specified queue on the current device for output
    /// within the scope of the body
    /// - Parameters:
    ///  - queue: the queue on the device to use
    ///  - body: a closure where the device queue will be used
    @inlinable func using<R>(queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        queueStack.append(validQueue(currentQueue.deviceIndex, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }

    // peforms a mod on the indexes to guarantee they are mapped into bounds
    @inlinable func validQueue(
        _ deviceId: Int,
        _ queueId: Int
    ) -> Device.Queue {
        let device = devices[deviceId % devices.count]
        let qcount = device.queues.count
        if qcount == 0 {
            return Self.syncQueue
        } else {
            return device.queues[queueId % qcount]
        }
    }
    
    //--------------------------------------------------------------------------
    /// randomSeed
    /// - Note: Whenever obtained, the random seed is also updated so that
    /// future stateless random TensorFlow op executions will result
    /// in non-deterministic results.
    @inlinable static var randomSeed: RandomSeed {
        get {
            let seed = lastRandomSeed
            lastRandomSeed = (seed.0, seed.1 + 1)
            return seed
        }
        set { lastRandomSeed = newValue }
    }
    
    @inlinable static func createRandomNumberGenerator(
        using seed: RandomSeed? = nil
    ) -> AnyRandomNumberGenerator {
        let randomSeed = seed ?? Platform.randomSeed
        let generatorSeed = UInt64(msb: UInt32(bitPattern: randomSeed.op),
                                   lsb: UInt32(bitPattern: randomSeed.graph))
        return AnyRandomNumberGenerator(
            PhiloxRandomNumberGenerator(uint64Seed: generatorSeed))
    }

}

//==============================================================================
// queue API

/// useSyncQueue
/// specifies the application thread queue to be used for operator execution
@inlinable public func useSyncQueue() {
    platform.useSyncQueue()
}

/// usingSyncQueue(body:
/// specifies the application thread queue to be used for operator execution
/// withing the scope of the closure
@inlinable public func usingSyncQueue<R>(_ body: () -> R) -> R {
    platform.usingSyncQueue(body)
}

/// use(device:queue:
/// specifies the device queue to use for operator execution
@inlinable public func use(device: Int, queue: Int = 0) {
    platform.use(device: device, queue: queue)
}

/// using(device:queue:body:
/// specifies the device queue to use for operator execution
/// withing the scope of the closure
@inlinable public func using<R>(device: Int, queue: Int = 0, _ body: () -> R) -> R {
    platform.using(device: device, queue: queue, body)
}

/// using(queue:body:
/// specifies the queue on the current device to use for operator execution
/// withing the scope of the closure
@inlinable public func using<R>(queue: Int, _ body: () -> R) -> R {
    platform.using(queue: queue, body)
}

/// testEachDevice(body:
/// executes `body` on each type of device for test coverage
@inlinable public func testEachDevice(_ onlyId: Int, _ body: () -> Void) {
    using(device: onlyId, body)
}

@inlinable public func testEachDevice(_ body: () -> Void) {
    usingSyncQueue(body)
    for i in 0..<platform.devices.count {
        using(device: i, body)
    }
}


//==============================================================================
/// the type used for memory indexing on discrete devices
public typealias DeviceIndex = Int32

//==============================================================================
// assert messages
@usableFromInline let _messageTensorShapeMismatch = "tensor shape mismatch"
@usableFromInline let _messageTensorOrderMismatch = "tensor order mismatch"
@usableFromInline let _messageElementsMustBeContiguous = "elements must be contigous"
@usableFromInline let _messageRepeatingStorageOrderNotSupported =
    "repeating storage order is not supported"

//==============================================================================
/// NanPropagation
public enum NanPropagation: Int, Codable {
    case propagate, noPropagate
}

//==============================================================================
/// ReductionType
public enum ReductionType: Int, Codable {
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
/// PlatformError
/// platform errors
public enum PlatformError : Error {
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
/// a compute device represents a physical platform device (cpu, gpu, ...)
public protocol ComputeDevice: class, Logging {
    associatedtype Queue: DeviceQueue
    
    /// the index of the device in the `devices` collection
    var index: Int { get }
    /// name used for diagnostics
    var name: String { get }
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }
    /// specifies the type of device memory for data transfer
    var memoryType: MemoryType { get }
    /// the number of queues available for execution. The user can set
    /// this value to change the number of queues available for execution.
}

//==============================================================================
/// DeviceMemory
public protocol DeviceMemory: class, Logging {
    /// base address and size of buffer
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// index of device where memory is located
    var deviceIndex: Int { get }
    /// mutable raw pointer to memory buffer to simplify driver calls
    var mutablePointer: UnsafeMutableRawPointer { get }
    /// the diagnostic name of the memory
    var name: String? { get set }
    /// raw pointer to memory buffer to simplify driver calls
    var pointer: UnsafeRawPointer { get }
    /// optional string for diagnostics
    var releaseMessage: String? { get set }
    /// specifies the device memory type for data transfer
    var type: MemoryType { get }
    /// version
    var version: Int { get set }
}

extension DeviceMemory {
    @inlinable public func count<E>(of type: E.Type) -> Int {
        buffer.count / MemoryLayout<E>.size
    }

    /// the number of bytes in the buffer
    @inlinable public var byteCount: Int { buffer.count }
}

//==============================================================================
//
public protocol QueueEvent: class {
    func signal()
    func wait()
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
// assert messages
@usableFromInline let _messageQueueThreadViolation =
"a queue can only be accessed by the thread that created it"

//==============================================================================
/// DeviceError
public enum DeviceError : Error {
    case allocationFailed
    case initializeFailed
    case queueError(idPath: [Int], message: String)
    case timeout(idPath: [Int], message: String)
}
