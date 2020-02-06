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
// Platform Abstraction Protocols
//
//  ComputePlatform
//      services[]
//        ComputeService (cpu, cuda, vulkan, ...)
//          devices[]
//            ComputeDevice (dev:0, dev:1, ...)
//              DeviceArray
//              DeviceQueue
//                CommandBuffer
//                QueueEvent
//
import Foundation

//==============================================================================
/// ComputePlatform
/// The compute platform is the root object for managing all services, devices,
/// and queues. There is one local instance per process, and possibly
/// many remote instances.
public protocol ComputePlatform: DeviceErrorHandling, ObjectTracking, Logger {
    /// global shared instance
    static var local: Platform { get }
    
    // instance members
    /// a device automatically selected based on service priority
    var defaultDevice: ComputeDevice { get }
    /// ordered list of device ids specifying the order for auto selection
    var deviceIdPriority: [Int] { get set }
    /// the platform id. Usually zero, but can be assigned in case a higher
    /// level object (e.g. cluster) will maintain a platform collection
    var id: Int { get set }
    /// the root logWriter
    var logWriter: Log { get set }
    /// location of dynamically loaded service modules
    var serviceModuleDirectory: URL { get set }
    /// ordered list of service names specifying the order for auto selection
    var servicePriority: [String] { get set }
    /// a dynamically loaded collection of available compute services.
    /// The "cpu" service will always be available
    var services: [String : ComputeService] { get }
    
    //--------------------------------------------------------------------------
    /// requestDevice(serviceName:deviceId:
    /// - Parameter serviceName: optional (cpu, cuda, tpu, ...)
    /// - Parameter deviceId: selected device id (0, 1, 2, ...)
    /// - Returns: the requested device from the requested service. If the
    /// service or device is not available, then a substrituion will be made
    /// based on Platform `servicePriority`and `deviceIdPriority`. The CPU
    /// is always available.
    func requestDevice(serviceName: String?, deviceId: Int) -> ComputeDevice
}

//==============================================================================
/// ComputeService
/// a compute service represents category of installed devices on the platform,
/// such as (cpu, cuda, tpu, ...)
public protocol ComputeService: ObjectTracking, Logger, DeviceErrorHandling {
    /// a collection of available devices
    var devices: [ComputeDevice] { get }
    /// the service id
    var id: Int { get }
    /// the service name used for `servicePriority` and logging
    var name: String { get }
    /// the platform this service belongs to
    var platform: ComputePlatform! { get }
    /// The default maximum amount of time allowed for an operation to complete.
    /// `timeout` is inherited by devices and queues when they are created.
    var timeout: TimeInterval? { get set }

    /// required initializer to support dynamically loaded services
    /// - Parameter platform: the parent platform object
    /// - Parameter id: the service id
    /// - Parameter logInfo: the log information to use
    /// - Parameter name: an optional service name
    init(platform: ComputePlatform,
         id: Int,
         logInfo: LogInfo,
         name: String?) throws
}

//==============================================================================
/// ServiceError
/// errors thrown from a ComputeService
public enum ServiceError : Error {
    case serviceIsUnavailable
    case functionFailure(location: String, message: String)
    case rangeError(String)
}

//==============================================================================
/// a set of predefined property names to simplify configuring
/// the service properties
public let cpuSynchronousServiceName = "cpuSync"
public let cpuAsynchronousServiceName = "cpuAsync"
public let testCpuServiceName = "testCpu"

//==============================================================================
public enum EvaluationMode {
    /// operation is used to perform inference
    case inferring
    /// operation is used to perform training
    case training
}

//==============================================================================
/// LocalComputeService
public protocol LocalComputeService: ComputeService { }
public protocol CpuServiceProtocol {}


public extension LocalComputeService {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    @inlinable
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            platform.handleDevice(error: error)
        }
    }
}

//==============================================================================
/// ComputeDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ComputeDevice: ObjectTracking, Logger, DeviceErrorHandling {
    //-------------------------------------
    // properties
    /// a key to lookup a TensorArray's DeviceArray replica associated with
    /// this device
    var deviceArrayReplicaKey: Int { get }
    /// describes the devices memory properties and available heaps
    var memory: MemoryProperties { get }
    /// parameters defining maximum device capabilties
    var limits: DeviceLimits { get }
    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// the name of the device
    var name: String { get }
    /// the service this device belongs to
    var service: ComputeService! { get }
    /// the maximum amount of time allowed for an operation to complete
    var timeout: TimeInterval? { get set }

    //-------------------------------------
    /// a collection of device queues that can be used for computation
    var queues: [DeviceQueue] { get }
    
    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    /// - Parameter byteCount: the number of bytes to allocate on the device
    /// - Parameter heapIndex: the index of the heap to use
    /// - Parameter zero: `true` to inialize the array to zero
    func createArray(byteCount: Int, heapIndex: Int, zero: Bool) throws
        -> DeviceArray
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
/// MemoryAttributes
// TODO: get and reword descriptions so that they make sense in our context.
// Some of these types maybe be eliminated if they are doing managed memory
// schemes by mappi the device memory into the host virtual address space.
// This mechanism is convenient but at least on Cuda has very poor performance
// and explicit memory transfers are much faster.
// https://vulkan.lunarg.com/doc/view/latest/windows/apispec.html#VkMemoryPropertyFlagBits
public struct MemoryAttributes: OptionSet, CustomStringConvertible {
    public let rawValue: Int

    /// this type is the most efficient for local device access
    public static let deviceLocal     = MemoryAttributes(rawValue: 1 << 0)
    public static let deviceCoherent  = MemoryAttributes(rawValue: 1 << 1)
    public static let deviceUncached  = MemoryAttributes(rawValue: 1 << 2)
    /// this type can be mapped for host access
    public static let hostVisible     = MemoryAttributes(rawValue: 1 << 3)
    /// this type specifies that the host and device share unified memory
    /// and no host cache management commands are required for transfer
    public static let hostCoherent    = MemoryAttributes(rawValue: 1 << 4)
    public static let hostCached      = MemoryAttributes(rawValue: 1 << 5)
    public static let lazilyAllocated = MemoryAttributes(rawValue: 1 << 6)
    public static let protected       = MemoryAttributes(rawValue: 1 << 7)
    
    @inlinable
    public var description: String {
        var string = "["
        if self.contains(.deviceLocal)     { string += ".deviceLocal, " }
        if self.contains(.hostVisible)     { string += ".hostVisible, " }
        if self.contains(.hostCoherent)    { string += ".hostCoherent, " }
        if self.contains(.hostCached)      { string += ".hostCached, " }
        if self.contains(.lazilyAllocated) { string += ".lazilyAllocated, " }
        if self.contains(.protected)       { string += ".protected, " }
        if self.contains(.deviceCoherent)  { string += ".deviceCoherent, "}
        if self.contains(.deviceUncached)  { string += ".deviceUncached, "}
        string.removeLast(2)
        string += "]"
        return string
    }

    //--------------------------------------------------------------------------
    @inlinable
    public init(rawValue: Int) { self.rawValue = rawValue }
}

//==============================================================================
/// DeviceMemoryProperties
public struct MemoryProperties {
    /// specifies if device memory is unified with host cpu or discreet
    // TODO: this will need to be reexamined when Vulan is added, because
    // some heaps may be host coherent and some not. For now the device
    // is either a uma device or not
    public var addressing: MemoryAddressing
    /// collection of device heaps
    public var heaps: [MemoryHeap]
}

//==============================================================================
/// MemoryAddressing
public enum MemoryAddressing {
    case unified, discreet
}

//==============================================================================
/// MemoryHeap
public struct MemoryHeap {
    /// total memory size in bytes
    public let size: Int
    /// a set of flags describing the heap attributes
    public let attributes: MemoryAttributes
    
    /// returns a current estimate of memory used and available in this heap
    @inlinable
    func budget() throws -> MemoryBudget {
        // TODO
        MemoryBudget(available: 0, used: 0)
    }
}

//==============================================================================
/// MemoryBudget
public struct MemoryBudget {
    /// a rough estimate of how much memory the process can allocate from
    /// the associated heap before allocations may fail or cause
    /// performance degradation
    public var available: Int
    /// an estimate of how much memory the process is currently using
    /// in the associated heap
    public var used: Int
    
    @inlinable
    public init(available: Int, used: Int) {
        self.available = available
        self.used = used
    }
}

//==============================================================================
/// LocalComputeDevice
public protocol LocalComputeDevice: ComputeDevice { }

public extension LocalComputeDevice {
    //--------------------------------------------------------------------------
    /// handleDevice(error:
    @inlinable
    func handleDevice(error: Error) {
        if (deviceErrorHandler?(error) ?? .propagate) == .propagate {
            service.handleDevice(error: error)
        }
    }
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking {
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the device that created this array
    var device: ComputeDevice { get }
    /// `true` if the array is read only
    var isReadOnly: Bool { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }
}

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `ComputeDevice`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent: ObjectTracking {
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

public struct QueueEventOptions: OptionSet {
    public let rawValue: Int
    public static let timing       = QueueEventOptions(rawValue: 1 << 0)
    public static let interprocess = QueueEventOptions(rawValue: 1 << 1)

    //--------------------------------------------------------------------------
    @inlinable
    public init() { self.rawValue = 0 }

    @inlinable
    public init(rawValue: Int) { self.rawValue = rawValue }
}

public enum QueueEventError: Error {
    case timedOut
}
