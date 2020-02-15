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

//==============================================================================
/// Platform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class Platform<Service>: ComputePlatformType
    where Service: PlatformServiceType
{
    // properties
    public let logInfo: LogInfo
    public let name: String
    public let service: Service
    public var queueStack: [(device: Int, queue: Int)]

    //--------------------------------------------------------------------------
    @inlinable
    public init(log: Log? = nil, name: String? = nil) {
        self.name = name ?? "platform"
        logInfo = LogInfo(logWriter: log ?? Current.log,
                          logLevel: .error,
                          namePath: self.name,
                          nestingLevel: 0)
        
        // create the service
        self.service = Service(parent: logInfo, id: 0)
        
        // selecting device 1 should be the first accelerated device
        // if there is only one device, then the index will wrap to zero
        self.queueStack = []
        self.queueStack = [ensureValidIndexes(1, 0)]
    }
}

//==============================================================================
/// ComputePlatform
/// The root collection of compute resources available to the application
/// on a given machine
public protocol ComputePlatform: Logger {
    /// a device queue whose memory is shared with the application
    var applicationQueue: DeviceQueue { get }
    /// the currently selected device queue to direct work
    /// - Returns: the current device queue
    var currentQueue: DeviceQueue { get }
    /// name used logging
    var name: String { get }
    /// the current device and queue to direct work
    var queueStack: [(device: Int, queue: Int)] { get set }
    /// the platform compute service existential
    var platformService: PlatformService { get }

    //-------------------------------------
    /// returns the selected compute device
    func device(_ id: Int) -> ServiceDevice
    /// mods the specified indices to ensure they select valid objects
    func ensureValidIndexes(_ device: Int, _ queue: Int) -> (Int, Int)
}

//------------------------------------------------------------------------------
//
@inlinable public func useCpu() { Current.platform.useCpu() }
@inlinable public func use(device: Int, queue: Int = 0) {
    Current.platform.use(device: device, queue: queue)
}
@inlinable public func using<R>(device: Int, queue: Int = 0,
                                _ body: () -> R) -> R {
    Current.platform.using(device: device, queue: queue, body)
}
@inlinable public func using<R>(queue: Int, _ body: () -> R) -> R {
    Current.platform.using(queue: queue, body)
}

// Platform extensions
extension ComputePlatform {
    /// changes the current device/queue to use cpu:0
    @inlinable
    public mutating func useCpu() {
        queueStack[queueStack.count - 1] = (0, 0)
    }
    /// selects the specified device queue for output
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    @inlinable
    public mutating func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = ensureValidIndexes(device, queue)
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
        queueStack.append(ensureValidIndexes(device, queue))
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
        queueStack.append(ensureValidIndexes(queueStack.last!.device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
}

//==============================================================================
/// ComputePlatformType
/// The root collection of compute resources available to the application
/// on a given machine
public protocol ComputePlatformType: ComputePlatform {
    // types
    associatedtype Service: PlatformServiceType
    
    // generic typed properties
    /// the local device compute service
    var service: Service { get }
}

//==============================================================================
/// ComputePlatformType extensions for queue stack manipulation
extension ComputePlatformType {
    @inlinable
    public var platformService: PlatformService { service }
    /// the currently active queue that API functions will use
    /// - Returns: the current device queue
    @inlinable
    public var currentDevice: ServiceDevice {
        service.devices[queueStack.last!.device]
    }
    /// returns the specified compute device
    /// - Returns: the current device queue
    @inlinable
    public func device(_ id: Int) -> ServiceDevice {
        service.devices[id]
    }
    /// the currently active queue that API functions will use
    /// - Returns: the current device queue
    @inlinable
    public var currentQueue: DeviceQueue {
        let (device, queue) = queueStack.last!
        return service.devices[device].queues[queue]
    }
    @inlinable
    public var applicationQueue: DeviceQueue {
        // TODO: add check to use current queue if it has unified memory
        // return cpu device queue for now
        service.devices[0].queues[0]
    }
    // peforms a mod on the indexes to guarantee they are mapped into bounds
    @inlinable
    public func ensureValidIndexes(_ device: Int, _ queue: Int) -> (Int, Int){
        let deviceIndex = device % service.devices.count
        let queueIndex = queue % service.devices[deviceIndex].queues.count
        return (deviceIndex, queueIndex)
    }
}

