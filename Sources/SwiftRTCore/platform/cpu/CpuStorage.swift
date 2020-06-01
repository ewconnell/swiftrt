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
    // peroperties
    public let alignment: Int
    public let byteCount: Int
    public let hostBuffer: UnsafeMutableRawBufferPointer
    public let id: Int
    public let isReadOnly: Bool
    public let isReference: Bool
    public let layout: Layout
    public var name: String
    
    @inlinable public func countOf<E: StorageElement>(type: E) -> Int {
        assert(byteCount % MemoryLayout<E>.size == 0,
               "Buffer size is not even multiple of Element type")
        return byteCount / MemoryLayout<E>.size
    }

    //--------------------------------------------------------------------------
    // init(type:count:layout:
    @inlinable public init<E: StorageElement>(
        type: E.Type,
        count: Int,
        layout: Layout
    ) {
        self.layout = layout
        alignment = MemoryLayout<E.Stored>.alignment
        byteCount = MemoryLayout<E.Stored>.size * E.storedCount(count)
        hostBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: self.byteCount,
            alignment: alignment)
        id = Context.nextBufferId
        isReadOnly = false
        isReference = false
        name = "Tensor"

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
            "\(E.self)[\(count)]", categories: .dataAlloc)
        #endif
    }

    //--------------------------------------------------------------------------
    // init(type:element:
    @inlinable public init<E: StorageElement>(
        type: E.Type,
        single element: E.Value
    ) {
        layout = .row
        alignment = MemoryLayout<E.Stored>.alignment
        byteCount = MemoryLayout<E.Stored>.size
        hostBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: self.byteCount,
            alignment: alignment)
        id = Context.nextBufferId
        isReadOnly = false
        isReference = true
        name = "Tensor"

        setElement(type: E.self, value: element, at: 0)

        #if DEBUG
        diagnostic("\(createString) \(diagnosticName) " +
            "\(E.self)[1]", categories: .dataAlloc)
        #endif
    }

    //--------------------------------------------------------------------------
    // init(other:
    @inlinable public init(copying other: CpuStorage) {
        alignment = other.alignment
        layout = other.layout
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
    // init(buffer:layout:
    @inlinable public init<E: StorageElement>(
        type: E.Type,
        referenceTo buffer: UnsafeBufferPointer<E.Stored>,
        layout: Layout
    ) {
        self.layout = layout
        alignment = MemoryLayout<E.Stored>.alignment
        byteCount = MemoryLayout<E.Stored>.size * buffer.count
        let buff = UnsafeMutableBufferPointer(mutating: buffer)
        self.hostBuffer = UnsafeMutableRawBufferPointer(buff)
        self.id = Context.nextBufferId
        self.isReadOnly = true
        self.isReference = true
        self.name = "Tensor"

        #if DEBUG
        diagnostic("\(createString) Reference \(diagnosticName) " +
            "\(E.self)[\(buffer.count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(type:buffer:layout:
    @inlinable public init<E: StorageElement>(
        type: E.Type,
        referenceTo buffer: UnsafeMutableBufferPointer<E.Stored>,
        layout: Layout
    ) {
        self.layout = layout
        alignment = MemoryLayout<E.Stored>.alignment
        byteCount = MemoryLayout<E.Stored>.size * buffer.count
        self.hostBuffer = UnsafeMutableRawBufferPointer(buffer)
        self.id = Context.nextBufferId
        self.isReadOnly = false
        self.isReference = true
        self.name = "Tensor"

        #if DEBUG
        diagnostic("\(createString) Reference \(diagnosticName) " +
            "\(E.self)[\(buffer.count)]", categories: .dataAlloc)
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
    @inlinable public func element<E: StorageElement>(
        type: E.Type,
        at index: Int
    ) -> E.Value {
        let typedBuffer = hostBuffer.bindMemory(to: E.Stored.self)
        return E.value(at: index, from: typedBuffer[E.storedIndex(index)])
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func setElement<E: StorageElement>(
        type: E.Type,
        value: E.Value,
        at index: Int
    ) {
        let typedBuffer = hostBuffer.bindMemory(to: E.Stored.self)
        E.store(value: value, at: index, to: &typedBuffer[E.storedIndex(index)])
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable public func read<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int
    ) -> UnsafeBufferPointer<E.Stored> {
        let raw = hostBuffer.baseAddress!
        let p = raw.bindMemory(to: E.Stored.self, capacity: count)
        return UnsafeBufferPointer(start: p.advanced(by: base), count: count)
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable public func read<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeBufferPointer<E.Stored> {
        let raw = hostBuffer.baseAddress!
        let p = raw.bindMemory(to: E.Stored.self, capacity: count)
        return UnsafeBufferPointer(start: p.advanced(by: base), count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable public func readWrite<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int
    ) -> UnsafeMutableBufferPointer<E.Stored> {
        let raw = hostBuffer.baseAddress!
        let p = raw.bindMemory(to: E.Stored.self, capacity: count)
        return UnsafeMutableBufferPointer(start: p.advanced(by: base), count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable public func readWrite<E: StorageElement>(
        type: E.Type,
        at base: Int,
        count: Int,
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableBufferPointer<E.Stored> {
        let raw = hostBuffer.baseAddress!
        let p = raw.bindMemory(to: E.Stored.self, capacity: count)
        return UnsafeMutableBufferPointer(start: p.advanced(by: base), count: count)
    }
}

