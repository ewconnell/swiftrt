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
import Foundation

//==============================================================================
/// TensorStorageProtocol
public protocol TensorStorageProtocol
{
    associatedtype Element
    
    /// the id returned from `createBuffer`
    var storage: BufferId { get }
    
    /// read(buffer:on:
    /// - Parameter id: id of the buffer
    /// - Parameter using: specifies the device/queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a buffer pointer to the stored elements
    func read(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeBufferPointer<Element>
    /// readWrite(buffer:on:
    /// - Parameter id: id of the buffer
    /// - Parameter using: specifies the device queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a mutable buffer pointer to the stored elements
    func readWrite(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeMutableBufferPointer<Element>
    /// overwrite(buffer:on:
    /// This function will be higher performance than `readWrite` if it is
    /// known that all elements will be written to, because it does not
    /// need to synchronize.
    /// - Parameter id: id of the buffer
    /// - Parameter using: specifies the device queue for synchronization.
    /// A value of `nil` blocks the caller until synchronization is complete.
    /// - Returns: a mutable buffer pointer to the stored elements.
    /// The data will not be synchronized and it is required that
    /// the operation will overwrite all elements of the buffer
    func overwrite(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeMutableBufferPointer<Element>
}

//==============================================================================
/// TensorStorageProtocol
public extension TensorStorageProtocol
{
    func read(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeBufferPointer<Element>
    {
        Current.service.read(storage, using: deviceQueue)
            .bindMemory(to: Element.self)
    }
    
    func readWrite(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeMutableBufferPointer<Element>
    {
        Current.service.readWrite(storage, using: deviceQueue)
            .bindMemory(to: Element.self)
    }
    
    func overwrite(_ id: BufferId, using deviceQueue: (Int, Int))
        -> UnsafeMutableBufferPointer<Element>
    {
        Current.service.overwrite(storage, using: deviceQueue)
            .bindMemory(to: Element.self)
    }
}

//==============================================================================
/// TensorStorage
/// reference counted access to a host memory buffer managed by the platform
public class TensorStorage<Element>: TensorStorageProtocol {
    public let storage: BufferId

    public init(count: Int) {
        let byteCount = MemoryLayout<Element>.size * count
        storage = Current.service.createBuffer(byteCount: byteCount)
    }
    
    deinit {
        Current.service.release(storage)
    }
}
