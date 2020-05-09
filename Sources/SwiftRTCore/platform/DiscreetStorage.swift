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
/// DiscreetStorage
public final class DiscreetStorage<Element>: StorageBuffer
{
    /// the number of storage elements
    public let count: Int
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
    /// replicated device memory buffers
    public var replicas: [DeviceMemory?]

    //--------------------------------------------------------------------------
    // init(count:
    @inlinable public init(count: Int) {
        self.count = count
        hostBuffer = UnsafeMutableBufferPointer(start: nil, count: 0)
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false
        master = -1
        name = "Tensor"
        let deviceCount = Context.local.platform.devices.count
        replicas = [DeviceMemory?](repeating: nil, count: deviceCount)

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
                    "\(Element.self)[\(count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(element:
    @inlinable public init(single element: Element) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public init(copying other: DiscreetStorage<Element>) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public init(referenceTo buffer: UnsafeBufferPointer<Element>) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public init(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>
    ) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public init<S, Stream>(
        block shape: S,
        bufferedBlocks: Int,
        stream: Stream
    ) where S : TensorShape, Stream : BufferStream {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func element(at offset: Int) -> Element {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func setElement(value: Element, at offset: Int) {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func read(
        at offset: Int, count: Int
    ) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func read(
        at offset: Int,
        count: Int,
        using queue: DeviceQueue
    ) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite(
        at offset: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite(
        at offset: Int,
        count: Int,
        willOverwrite: Bool,
        using queue: DeviceQueue
    ) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    @inlinable func getMemory(_ queue: DeviceQueue) throws -> DeviceMemory {
        if let memory = replicas[queue.deviceId] {
            return memory
        } else {
            fatalError()
        }
    }
}
