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

//==============================================================================
/// SyncStorage
/// A synchronous host memory element storage buffer
public final class CpuStorage: StorageBuffer {
    // StorageBuffer protocol properties
    public let alignment: Int
    public let byteCount: Int
    public let id: Int
    public let isReadOnly: Bool
    public let isReference: Bool
    public var name: String

    //--------------------------------------------------------------------------
    // implementation properties

    // host storage buffer
    public let hostBuffer: UnsafeMutableRawBufferPointer

    //--------------------------------------------------------------------------
    // init(type:count:name:
    @inlinable public init<Element>(
        storedType: Element.Type,
        count: Int,
        name: String
    ) {
        assert(MemoryLayout<Element>.size != 0,
               "type: \(Element.self) is size 0")
        self.name = name
        alignment = MemoryLayout<Element>.alignment
        byteCount = MemoryLayout<Element>.size * count
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false

        hostBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount,
            alignment: alignment)

        #if DEBUG
        diagnostic("\(allocString) \(diagnosticName) " +
            "\(Element.self)[\(count)]", categories: .dataAlloc)
        #endif
    }

    //--------------------------------------------------------------------------
    // init(other:queue:
    @inlinable public init(copying other: CpuStorage, using queue: DeviceQueue){
        alignment = other.alignment
        byteCount = other.byteCount
        id = Context.nextBufferId
        isReadOnly = other.isReadOnly
        isReference = other.isReference
        name = other.name
        if isReference {
            hostBuffer = other.hostBuffer
        } else {
            hostBuffer = UnsafeMutableRawBufferPointer.allocate(
                byteCount: other.byteCount,
                alignment: other.alignment)
            hostBuffer.copyMemory(from: UnsafeRawBufferPointer(other.hostBuffer))
        }
    }

    //--------------------------------------------------------------------------
    // init(buffer:
    @inlinable public init<Element>(
        referenceTo buffer: UnsafeBufferPointer<Element>
    ) {
        alignment = MemoryLayout<Element>.alignment
        byteCount = MemoryLayout<Element>.size * buffer.count
        let buff = UnsafeMutableBufferPointer(mutating: buffer)
        self.hostBuffer = UnsafeMutableRawBufferPointer(buff)
        self.id = Context.nextBufferId
        self.isReadOnly = true
        self.isReference = true
        self.name = "Tensor"

        #if DEBUG
        diagnostic("\(referenceString) \(diagnosticName) " +
            "\(Element.self)[\(buffer.count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(type:buffer:
    @inlinable public init<Element>(
        referenceTo buffer: UnsafeMutableBufferPointer<Element>
    ) {
        alignment = MemoryLayout<Element>.alignment
        byteCount = MemoryLayout<Element>.size * buffer.count
        self.hostBuffer = UnsafeMutableRawBufferPointer(buffer)
        self.id = Context.nextBufferId
        self.isReadOnly = false
        self.isReference = true
        self.name = "Tensor"

        #if DEBUG
        diagnostic("\(referenceString) \(diagnosticName) " +
            "\(Element.self)[\(buffer.count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // streaming
    @inlinable
    public init<S, Stream>(block shape: S, bufferedBlocks: Int, stream: Stream)
        where S: TensorShape, Stream: BufferStream
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // deinit
    @inlinable deinit {
        if !isReference {
            hostBuffer.deallocate()
            #if DEBUG
            diagnostic("\(releaseString) \(diagnosticName) ",
                categories: .dataAlloc)
            #endif
        }
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable public func read<Element>(
        type: Element.Type,
        at index: Int,
        count: Int
    ) -> UnsafeBufferPointer<Element> {
        // advance to typed starting position
        let start = hostBuffer.baseAddress!
                .bindMemory(to: Element.self, capacity: count)
                .advanced(by: index)
        // return read only buffer pointer
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable public func read<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: DeviceQueue
    ) -> UnsafeBufferPointer<Element> {
        read(type: type, at: index, count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable public func readWrite<Element>(
        type: Element.Type,
        at index: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<Element> {
        // advance to typed starting position
        let start = hostBuffer.baseAddress!
                .bindMemory(to: Element.self, capacity: count)
                .advanced(by: index)
        // return read/write buffer pointer
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable public func readWrite<Element>(
        type: Element.Type,
        at index: Int,
        count: Int,
        using queue: DeviceQueue
    ) -> UnsafeMutableBufferPointer<Element> {
        readWrite(type: type, at: index, count: count)
    }
}

