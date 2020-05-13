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
/// DiscreetStorage
public final class DiscreetStorage<Element>: StorageBuffer
{
    /// the number of storage elements
    public let count: Int
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
    public var replicas: [DeviceMemory<Element>?]

    /// the host transfer buffer
    @inlinable public var hostBuffer: UnsafeMutableBufferPointer<Element> {
        assert(master == 0, "`read` or `readWrite` on device 0" +
               " must be called prior to access")
        return replicas[0]!.buffer
    }

    //--------------------------------------------------------------------------
    // init(count:
    @inlinable public init(count: Int) {
        self.count = count
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false
        master = -1
        name = "Tensor"
        let count = Context.local.platform.devices.count
        replicas = [DeviceMemory<Element>?](repeating: nil, count: count)

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
                    "\(Element.self)[\(count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(element:
    @inlinable public convenience init(single element: Element) {
        self.init(count: 1)
        readWrite(at: 0, count: 1)[0] = element

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
                    "\(Element.self)[1]", categories: .dataAlloc)
        #endif
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
        read(at: offset, count: 1)[0]
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func setElement(value: Element, at offset: Int) {
        readWrite(at: offset, count: 1)[0] = value
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func read(
        at offset: Int, count: Int
    ) -> UnsafeBufferPointer<Element> {
        let queue = Context.cpuQueue(0)
        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func read(
        at offset: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeBufferPointer<Element> {
        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite(
        at offset: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<Element> {
        let queue = Context.cpuQueue(0)
        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func readWrite(
        at offset: Int,
        count: Int,
        willOverwrite: Bool,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableBufferPointer<Element> {
        let start = getMemory(queue).buffer.baseAddress!.advanced(by: offset)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // getMemory
    // Manages an array of replicated device memory indexed by the deviceId
    // assoicated with `stream`. It will lazily create device memory if needed
    @inlinable public func getMemory(
        _ queue: PlatformType.Device.Queue
    ) -> DeviceMemory<Element> {
        if let memory = replicas[queue.deviceId] {
            if memory.version == replicas[master]!.version {
                return memory
            } else {
                // migrate
                fatalError()
            }
        } else {
            do {
                // allocate the buffer for the target device
                // and save in the replica list
                let memory = try queue.allocate(Element.self, count: count)
                replicas[queue.deviceId] = memory
                
                // the new buffer is now the master version
                master = queue.deviceId
                return memory
            } catch {
                // Fail for now
                writeLog("Failed to allocate memory on \(queue.deviceName)")
                fatalError("TODO: implement LRU host migration" +
                            " and discreet memory discard")
            }
        }
    }
}
