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
/// DeviceMemory
public struct DeviceMemory {
    /// base address and size of buffer
    public let buffer: UnsafeMutableRawBufferPointer
    /// function to free the memory
    public let deallocate: () -> Void
    /// specifies the device memory type for data transfer
    public let memoryType: MemoryType
    /// version
    public var version: Int
    
    @inlinable public init(
        buffer: UnsafeMutableRawBufferPointer,
        memoryType: MemoryType,
        _ deallocate: @escaping () -> Void
    ) {
        self.buffer = buffer
        self.memoryType = memoryType
        self.version = -1
        self.deallocate = deallocate
    }
}

//==============================================================================
/// AsyncStorage
public final class AsyncStorage<Element>: StorageBuffer
{
    /// replicated device local memory buffers
    public var memory: [DeviceMemory?]
    /// the host transfer buffer
    public let hostBuffer: UnsafeMutableBufferPointer<Element>
    /// unique storage id used in diagnostic messages
    public let id: Int
    /// `true` if the storage is read only
    public let isReadOnly: Bool
    /// `true` if the storage is a reference to externally
    /// managed memory
    public let isReference: Bool
    /// the index of the last memory buffer written to
    public var master: Int
    /// the name of the storage used in diagnostic messages
    public var name: String


    
    //--------------------------------------------------------------------------
    // init(count:name:
    public init(count: Int) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // init(element:
    public init(single element: Element) {
        fatalError()
    }
    
    public init(copying other: AsyncStorage<Element>) {
        fatalError()
    }
    
    public init(referenceTo buffer: UnsafeBufferPointer<Element>) {
        fatalError()
    }
    
    public init(referenceTo buffer: UnsafeMutableBufferPointer<Element>) {
        fatalError()
    }
    
    public init<S, Stream>(block shape: S, bufferedBlocks: Int, stream: Stream) where S : TensorShape, Stream : BufferStream {
        fatalError()
    }
    
    public func element(at offset: Int) -> Element {
        fatalError()
    }
    
    public func setElement(value: Element, at offset: Int) {
        fatalError()
    }
    
    public func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    public func read(at offset: Int, count: Int, using queue: DeviceQueue) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    public func readWrite(at offset: Int, count: Int) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
    
    public func readWrite(at offset: Int, count: Int, willOverwrite: Bool, using queue: DeviceQueue) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    private func getMemory(_ stream: DeviceQueue) throws -> DeviceMemory {
        fatalError()
    }
}
