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


// REMOVE THIS HACK!!
public func copy<T, U>(from src: T, to dest: inout U)
    where T: TensorType, U: MutableTensorType, T.Element == U.Element
{
    zip(dest.indices, src).forEach { dest[$0] = $1 }
}


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
    /// the current device and queue to direct work
    var queueStack: [QueueId] { get set }
}

public extension Platform {
    /// the currently active device that platform functions will use
    /// - Returns: the current device
    @inlinable @_transparent
    var currentDevice: Device {
        devices[currentQueueId.device]
    }

    /// the currently active queue that platform functions will use
    /// - Returns: the current device queue
    @inlinable @_transparent
    var currentQueue: DeviceQueue {
        let queueId = currentQueueId
        return devices[queueId.device].queues[queueId.queue]
    }

    /// the currently active queue that platform functions will use
    /// - Returns: the current device queue
    @inlinable @_transparent
    var currentQueueId: QueueId { queueStack.last! }
}

//==============================================================================
// queue API
@inlinable public func useCpu() {
    Context.platform.useCpu()
}

@inlinable public func use(device: Int, queue: Int = 0) {
    Context.platform.use(device: device, queue: queue)
}

@inlinable public func using<R>(device: Int, queue: Int = 0,
                                _ body: () -> R) -> R {
    Context.platform.using(device: device, queue: queue, body)
}

@inlinable public func using<R>(queue: Int, _ body: () -> R) -> R {
    Context.platform.using(queue: queue, body)
}

// Platform extensions
public extension Platform {
    /// changes the current device/queue to use cpu:0
    @inlinable
    func useCpu() {
        queueStack[queueStack.count - 1] = QueueId(0, 0)
    }
    /// selects the specified device queue for output
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    @inlinable
    func use(device: Int, queue: Int = 0) {
        queueStack[queueStack.count - 1] = ensureValidId(device, queue)
    }
    /// selects the specified device queue for output within the scope of
    /// the body
    /// - Parameter device: the device to use. Device 0 is the cpu
    /// - Parameter queue: the queue on the device to use
    /// - Parameter body: a closure where the device queue will be used
    @inlinable
    func using<R>(device: Int, queue: Int = 0, _ body: () -> R) -> R {
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
    func using<R>(queue: Int, _ body: () -> R) -> R {
        // push the selection onto the queue stack
        let current = queueStack.last!
        queueStack.append(ensureValidId(current.device, queue))
        defer { _ = queueStack.popLast() }
        return body()
    }
    // peforms a mod on the indexes to guarantee they are mapped into bounds
    @inlinable
    func ensureValidId(_ deviceId: Int, _ queueId: Int) -> QueueId {
        let device = deviceId % devices.count
        let queue = queueId % devices[device].queues.count
        return QueueId(device, queue)
    }
}

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

