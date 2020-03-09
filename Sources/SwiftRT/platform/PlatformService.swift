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
/// PlatformService
/// a compute service represents a category of installed devices on the
/// platform, such as (cpu, cuda, tpu, ...)
public protocol PlatformService: class {
    // types
    associatedtype Device: ServiceDevice

    /// a collection of available compute devices
    var devices: [Device] { get }
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
        let queueId = currentQueueId
        return devices[queueId.device].queues[queueId.queue]
    }

    /// the currently active queue that platform service functions will use
    /// - Returns: the current device queue
    @inlinable
    var currentQueueId: QueueId { queueStack.last! }

    //--------------------------------------------------------------------------
    /// read(tensor:
    /// gains synchronized read only access to the tensor `bufferRef`
    /// - Parameter tensor: the tensor to read
    /// - Returns: an `BufferElements` that can be used to iterate the shape
    @inlinable
    func read<T>(_ tensor: T) -> BufferElements<T.Element, T.Shape>
        where T: TensorView
    {
        BufferElements(tensor.shape, tensor.read(using: currentQueue))
    }

    //--------------------------------------------------------------------------
    /// write(tensor:willOverwrite:
    /// gains synchronized read write access to the tensor `bufferRef`
    /// - Parameter tensor: the tensor to read
    /// - Parameter willOverwrite: `true` if all elements will be written
    /// - Returns: an `BufferElements` that can be used to iterate the shape
    @inlinable
    func write<T>(_ tensor: inout T, willOverwrite: Bool = true)
        -> MutableBufferElements<T.Element, T.Shape> where T: TensorView
    {
        // get the write buffer
        let buffer = tensor.readWrite(willOverwrite: willOverwrite,
                                      using: currentQueue)
        
        // return a mutable shaped buffer iterator
        return MutableBufferElements(tensor.shape, buffer)
    }

    //--------------------------------------------------------------------------
    /// `createResult(shape:name:`
    /// creates a new tensor like the one specified and access to it's
    /// `bufferRef`
    /// - Parameter other: a tensor to use as a template
    /// - Parameter shape: the shape of the tensor to create
    /// - Parameter name: an optional name for the new tensor
    /// - Returns: a tensor and an associated `MutableBufferElements`
    /// that can be used to iterate the shape
    @inlinable
    func createResult<T>(like other: T, with shape: T.Shape, name: String? = nil)
        -> (T, MutableBufferElements<T.Element, T.Shape>) where T: TensorView
    {
        var result = other.createDense(with: shape, name: name)
        let resultBuffer = write(&result)
        assert(result.isUniquelyReference())
        return (result, resultBuffer)
    }

    //--------------------------------------------------------------------------
    /// `createResult(other:name:`
    /// creates a new tensor like the one specified and access to it's
    /// `bufferRef`
    /// - Parameter other: a tensor to use as a template
    /// - Parameter name: an optional name for the new tensor
    /// - Returns: a tensor and an associated `MutableBufferElements`
    /// that can be used to iterate the shape
    @inlinable
    func createResult<T>(like other: T, name: String? = nil)
        -> (T, MutableBufferElements<T.Element, T.Shape>) where T: TensorView
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
public extension PlatformService {
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

public typealias ReduceOpFinal<R: MutableShapedBuffer> = (R.Element) -> R.Element

//==============================================================================
// parameter matching helper
@inlinable
public func implicitlyMatchExtents<T>(_ lhs: T, _ rhs: T) -> (T, T)
    where T: TensorView
{
    if lhs.count == rhs.count {
        return (lhs, rhs)
    } else if lhs.count > rhs.count {
        return (lhs, rhs.repeated(to: lhs.bounds))
    } else {
        return (lhs.repeated(to: rhs.bounds), rhs)
    }
}

