////******************************************************************************
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

//==============================================================================
/// PlatformService
/// a compute service represents a category of installed devices on the
/// platform, such as (cpu, cuda, tpu, ...)
public protocol PlatformService: ServiceMemoryManagement, Logger {
    // types
    associatedtype Device: ServiceDeviceType
    
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

public typealias BufferHandle = Int

//==============================================================================
/// ServiceMemoryManagement
public protocol ServiceMemoryManagement {
    /// a collection of device buffer dictionaries indexed by the device
    /// number, and keyed by the handle returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [[BufferHandle : BufferDescription]] { get set }
    /// a dictionary relating a buffer handle to which device has the
    /// most recently mutated version. This is updated each time a write
    /// buffer is obtained on a different device
    /// - Parameter key: the buffer handle
    /// - Parameter value: the index of the device that has the master version
    var masterVersion: [BufferHandle : Int] { get set }

    //--------------------------------------------------------------------------
    /// createBuffer(byteCount:
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// Handles are used because the associated memory can be moved by the
    /// platform between accesses in order to maximize memory utilization.
    /// - Parameter byteCount: the size of the associated buffer in bytes
    /// suitably aligned for any type
    /// - Returns: a handle used to reference the buffer
    func createBuffer(byteCount: Int) -> BufferHandle
    /// createReference(to:
    /// creates a platform buffer entry whose storage is associated with
    /// the specified buffer pointer. No memory is allocated, so the
    /// buffer must point to valid storage space managed by the application.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter buffer: a buffer pointer to the data
    /// - Returns: a handle used to reference the buffer
    func createReference(to buffer: UnsafeRawBufferPointer) -> BufferHandle
    /// createMutableReference(to:
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Returns: a handle used to reference the buffer
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer)
        -> BufferHandle
    /// release(buffer:
    /// Releases a buffer created by calling `createBuffer`
    /// - Parameter buffer: the handle of the buffer to release.
    func release(buffer handle: BufferHandle)
    /// read(buffer:on:
    /// - Parameter buffer: handle to the buffer
    /// - Parameter on: specifies the device queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified handle. The data will be synchronized
    func read(buffer: BufferHandle,
              on: (device: Int, queue: Int)?) -> UnsafeRawBufferPointer
    /// readWrite(buffer:on:
    /// - Parameter buffer: handle to the buffer
    /// - Parameter on: specifies the device queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a mutable buffer pointer to the bytes associated with the
    /// specified handle. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite(buffer: BufferHandle, on: (device: Int, queue: Int)?)
        -> UnsafeMutableRawBufferPointer
    /// overwrite(buffer:on:
    /// This function will be higher performance than `readWrite` if it is
    /// known that all elements will be written to, because it does not
    /// need to synchronize.
    /// - Parameter buffer: handle to the buffer
    /// - Parameter on: specifies the device queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a mutable buffer pointer to the bytes associated with the
    /// specified handle. The data will not be synchronized and it is
    /// required that the operation will overwrite all elements of the buffer.
    func overwrite(buffer: BufferHandle, on: (device: Int, queue: Int)?)
        -> UnsafeMutableRawBufferPointer
}

//==============================================================================
// placeholder
public extension ServiceMemoryManagement {
    func createBuffer(byteCount: Int) -> BufferHandle { fatalError() }
    func createReference(to buffer: UnsafeRawBufferPointer) -> BufferHandle { fatalError() }
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer) -> BufferHandle  { fatalError() }
    func release(buffer handle: BufferHandle)  { fatalError() }
    func read(buffer: BufferHandle, on: (device: Int, queue: Int)?) -> UnsafeRawBufferPointer  { fatalError() }
    func readWrite(buffer: BufferHandle, on: (device: Int, queue: Int)?) -> UnsafeMutableRawBufferPointer  { fatalError() }
    func overwrite(buffer: BufferHandle, on: (device: Int, queue: Int)?) -> UnsafeMutableRawBufferPointer  { fatalError() }
}
