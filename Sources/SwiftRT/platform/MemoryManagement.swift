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
/// ServiceMemoryManagement
public protocol ServiceMemoryManagement {
    /// a collection of device buffer dictionaries indexed by the device
    /// number, and keyed by the id returned from `createDeviceBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [[Int : BufferDescription]] { get set }
    /// a dictionary relating a buffer id to which device has the
    /// most recently mutated version. This is updated each time a write
    /// buffer is obtained on a different device
    /// - Parameter key: the buffer id
    /// - Parameter value: the index of the device that has the master version
    var masterVersion: [Int : Int] { get set }
    
    //--------------------------------------------------------------------------
    /// createDeviceBuffer(byteCount:
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// Id(s) are used because the associated memory can be moved by the
    /// platform between accesses in order to maximize memory utilization.
    /// - Parameter type: the element type of the buffer
    /// - Parameter count: size of the buffer in `Element` units
    /// - Returns: a reference to the device buffer
    func createDeviceBuffer<Element>(of type: Element.Type, count: Int)
        -> BufferId

    /// cachedDeviceBuffer(element:
    /// returns a device buffer initialized with the specified `Element`
    /// value. User expressions use a lot of constant scalar values
    /// which are repeated. For example: `let m = matrix + 1`. These
    /// expressions are frequently iterated thousands of times. This function
    /// will maintain a cache of constant values, which are likely to
    /// already be present on a discreet accelerator device,
    /// saving a lot of time.
    /// - Parameter element: the element value to cache
    /// - Returns: a device buffer reference that contains the element value.
    /// A BufferId is created if it does not already exist.
    func cachedDeviceBuffer<Element>(for element: Element) -> BufferId

    /// createReference(to:
    /// creates a device buffer whose data is associated with
    /// the specified buffer pointer. No memory is allocated, so the
    /// buffer must point to valid data space managed by the application.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter applicationBuffer: a buffer pointer to the data
    /// - Returns: a reference to the device buffer
    func createReference<Element>(
        to applicationBuffer: UnsafeBufferPointer<Element>) -> BufferId

    /// createMutableReference(to:
    /// - Parameter applicationBuffer: a mutable buffer pointer to the data
    /// - Returns: a reference to the device buffer
    func createMutableReference<Element>(
        to applicationBuffer: UnsafeMutableBufferPointer<Element>) -> BufferId

    /// duplicate
    /// makes a duplicate of the specified device buffer. Used to support
    /// copy-on-write semantics
    /// - Parameter other: the id of the other device buffer to duplicate
    /// - Parameter queue: specifies the device/queue for synchronization.
    /// - Returns: a reference to the device buffer
    func duplicate(_ other: BufferId, using queue: QueueId) -> BufferId

    /// release(buffer:
    /// Releases a buffer created by calling `createDeviceBuffer`
    /// - Parameter buffer: the device buffer to release
    func release(_ buffer: BufferId)

    /// read(buffer:type:offset:queue:
    /// - Parameter buffer: the device buffer id to read
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter queue: device queue specification for data placement and
    /// synchronization. A value of `nil` will block the caller until the data
    /// is available in the application address space
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized
    func read<Element>(_ buffer: BufferId, of type: Element.Type, at offset: Int,
                       using queue: QueueId?) -> UnsafeBufferPointer<Element>

    /// readWrite(buffer:type:offset:queue:willOverwrite:
    /// - Parameter buffer: the device buffer id to readWrite
    /// - Parameter type: the element type of the buffer
    /// - Parameter offset: the offset in element sized units from
    /// the beginning of the buffer to read
    /// - Parameter queue: device queue specification for data placement and
    /// synchronization. A value of `nil` will block the caller until the data
    /// is available in the application address space
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the elements associated with the
    /// specified buffer id. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite<Element>(_ buffer: BufferId, of type: Element.Type,
                            at offset: Int, using queue: QueueId?,
                            willOverwrite: Bool)
        -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
/// BufferDescription
public struct BufferDescription {
    /// pointer to device buffer
    var buffer: UnsafeMutableRawBufferPointer
    /// `true` if the buffer can be mutated. The type of `buffer` is
    /// defined as mutable, but covers both cases to reduce generic complexity.
    let isMutable: Bool
    /// a buffer name used in diagnostic messages
    let name: String
    /// the mutation version of the buffer used for synchronization
    var version: Int
}

//==============================================================================
/// BufferId
/// a reference counted id for a service device buffer
public class BufferId {
    public let id: Int
    public init(_ id: Int) { self.id = id }
}

//==============================================================================
// placeholder
public extension ServiceMemoryManagement {
    func cachedDeviceBuffer<Element>(for element: Element) -> BufferId { fatalError() }
    func createDeviceBuffer<T>(of type: T.Type, count: Int) -> BufferId { fatalError() }
    func createReference<Element>(to applicationBuffer: UnsafeBufferPointer<Element>) -> BufferId { fatalError() }
    func createMutableReference<Element>(to applicationBuffer: UnsafeMutableBufferPointer<Element>) -> BufferId  { fatalError() }
    func duplicate(_ buffer: BufferId, using queue: QueueId) -> BufferId  { fatalError() }
    func release(_ buffer: BufferId)  { fatalError() }
    func read<T>(_ buffer: BufferId, of type: T.Type, at offset: Int, using queue: QueueId?) -> UnsafeBufferPointer<T> { fatalError() }
    func readWrite<T>(_ buffer: BufferId, of type: T.Type, at offset: Int, using queue: QueueId?, willOverwrite: Bool) -> UnsafeMutableBufferPointer<T> { fatalError() }
}
