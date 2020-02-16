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
    /// number, and keyed by the id returned from `createStorage`.
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
    /// cachedStorage(element:
    /// returns a device buffer initialized with the specified `Element`
    /// value. User expressions use a lot of constant scalar values
    /// which are repeated. For example: `let m = matrix + 1`. These
    /// expressions are frequently iterated thousands of times. This function
    /// will maintain a cache of constant values, which are likely to
    /// already be present on a discreet accelerator device,
    /// saving a lot of time.
    /// - Parameter element: the element value to cache
    /// - Returns: a reference to storage that contains the element value.
    /// Storage is created if it does not already exist.
    func cachedStorage<Element>(for element: Element) -> DeviceStorage
    /// createStorage(byteCount:
    /// creates a lazily allocated buffer to be used in tensor operations.
    /// Id(s) are used because the associated memory can be moved by the
    /// platform between accesses in order to maximize memory utilization.
    /// - Parameter byteCount: the size of the associated buffer in bytes
    /// suitably aligned for any type
    /// - Returns: a reference to storage
    func createStorage(byteCount: Int) -> DeviceStorage
    /// createReference(to:
    /// creates a platform buffer entry whose storage is associated with
    /// the specified buffer pointer. No memory is allocated, so the
    /// buffer must point to valid storage space managed by the application.
    /// This can be used to access things like hardware buffers or
    /// memory mapped files, network buffers, database results, without
    /// requiring an additional copy operation.
    /// - Parameter buffer: a buffer pointer to the data
    /// - Returns: a reference to storage
    func createReference(to buffer: UnsafeRawBufferPointer) -> DeviceStorage
    /// createMutableReference(to:
    /// - Parameter buffer: a mutable buffer pointer to the data
    /// - Returns: a reference to storage
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer)
        -> DeviceStorage
    /// duplicate
    /// makes a duplicate of the specified storage. Used to support
    /// copy-on-write semantics
    /// - Parameter storage: the storage id of the buffer to duplicate
    /// - Parameter queue: specifies the device/queue for synchronization.
    /// - Returns: a reference to storage
    func duplicate(_ storage: DeviceStorage, using queue: QueueId)
        -> DeviceStorage
    /// release(storage:
    /// Releases a buffer created by calling `createStorage`
    /// - Parameter storage: the device storage to release
    func release(_ storage: DeviceStorage)
    /// read(storage:queue:
    /// - Parameter storage: the device storage to read
    /// - Parameter queue: device queue specification for data placement and
    /// synchronization. A value of `nil` will block the caller until the data
    /// is available in the application address space
    /// - Returns: a buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized
    func read(_ storage: DeviceStorage, using queue: QueueId?)
        -> UnsafeRawBufferPointer
    /// readWrite(storage:queue:
    /// - Parameter storage: the device storage to read/write
    /// - Parameter queue: device queue specification for data placement and
    /// synchronization. A value of `nil` will block the caller until the data
    /// is available in the application address space
    /// - Parameter willOverwrite: `true` if the caller guarantees all
    /// buffer elements will be overwritten
    /// - Returns: a mutable buffer pointer to the bytes associated with the
    /// specified buffer id. The data will be synchronized so elements can be
    /// read before written, or sparsely written to
    func readWrite(_ storage: DeviceStorage, using queue: QueueId?,
                   willOverwrite: Bool) -> UnsafeMutableRawBufferPointer
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
/// DeviceStorage
/// a reference counted id for device storage
public class DeviceStorage {
    public let id: Int
    public init(_ id: Int) { self.id = id }
}

//==============================================================================
// placeholder
public extension ServiceMemoryManagement {
    func cachedStorage<Element>(for element: Element) -> DeviceStorage { fatalError() }
    func createStorage(byteCount: Int) -> DeviceStorage { fatalError() }
    func createReference(to buffer: UnsafeRawBufferPointer) -> DeviceStorage { fatalError() }
    func createMutableReference(to buffer: UnsafeMutableRawBufferPointer) -> DeviceStorage  { fatalError() }
    func duplicate(_ storage: DeviceStorage, using queue: QueueId) -> DeviceStorage  { fatalError() }
    func release(_ storage: DeviceStorage)  { fatalError() }
    func read(_ storage: DeviceStorage, using queue: QueueId?) -> UnsafeRawBufferPointer  { fatalError() }
    func readWrite(_ storage: DeviceStorage, using queue: QueueId?, willOverwrite: Bool) -> UnsafeMutableRawBufferPointer  { fatalError() }
}
