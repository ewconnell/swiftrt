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
    /// number, and keyed by the id returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [[BufferId : BufferDescription]] { get set }
    /// a dictionary relating a buffer id to which device has the
    /// most recently mutated version. This is updated each time a write
    /// buffer is obtained on a different device
    /// - Parameter key: the buffer id
    /// - Parameter value: the index of the device that has the master version
    var masterVersion: [BufferId : Int] { get set }
    
    //--------------------------------------------------------------------------
    /// createBuffer(byteCount:
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// Handles are used because the associated memory can be moved by the
    /// platform between accesses in order to maximize memory utilization.
    /// - Parameter byteCount: the size of the associated buffer in bytes
    /// suitably aligned for any type
    /// - Returns: an id used to reference the buffer
    func createBuffer(byteCount: Int) -> BufferId
    /// createReference(to:
    /// creates a platform buffer entry whose storage is associated with
    /// the specified buffer pointer. No memory is allocated, so the
    /// buffer must point to valid storage space managed by the application.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter buffer: a buffer pointer to the data
    /// - Returns: an id used to reference the buffer
    func createReference(to buffer: UnsafeRawBufferPointer) -> BufferId
    /// createMutableReference(to:
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Returns: an id used to reference the buffer
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer)
        -> BufferId
    /// duplicate
    /// makes a duplicate of the specified buffer. Used to support
    /// copy-on-write semantics
    /// - Parameter id: id of the buffer to duplicate
    /// - Parameter using: specifies the device/queue for synchronization.
    /// - Returns: id of the new buffer
    func duplicate(_ id: BufferId, using deviceQueue: (device: Int, queue: Int))
        -> BufferId
    /// release(id:
    /// Releases a buffer created by calling `createBuffer`
    /// - Parameter id: the id of the buffer to release
    func release(_ id: BufferId)
    /// read(buffer:on:
    /// - Parameter id: id of the buffer
    /// - Parameter using: specifies the device/queue for synchronization.
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized
    func read(_ id: BufferId, using deviceQueue: (device: Int, queue: Int))
        -> UnsafeRawBufferPointer
    /// readWrite(buffer:on:
    /// - Parameter id: id of the buffer
    /// - Parameter using: specifies the device queue for synchronization.
    /// - Parameter overwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite(_ id: BufferId, using deviceQueue: (device: Int, queue: Int),
                   overwrite: Bool)
        -> UnsafeMutableRawBufferPointer
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

public struct BufferId: Hashable {
    public let id: Int
    public init(_ id: Int) { self.id = id }
}

//==============================================================================
// placeholder
public extension ServiceMemoryManagement {
    func createBuffer(byteCount: Int) -> BufferId { fatalError() }
    func createReference(to buffer: UnsafeRawBufferPointer) -> BufferId { fatalError() }
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer) -> BufferId  { fatalError() }
    func duplicate(_ id: BufferId, using deviceQueue: (device: Int, queue: Int))
        -> BufferId  { fatalError() }
    func release(_ id: BufferId)  { fatalError() }
    func read(_ id: BufferId, using deviceQueue: (device: Int, queue: Int)) -> UnsafeRawBufferPointer  { fatalError() }
    func readWrite(_ id: BufferId, using deviceQueue: (device: Int, queue: Int),
                   overwrite: Bool) -> UnsafeMutableRawBufferPointer  { fatalError() }
}
