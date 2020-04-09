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
/// CpuBuffer
/// Used to manage a host memory buffer
public final class CpuBuffer<Element>: StorageBuffer
{
    public let hostBuffer: UnsafeMutableBufferPointer<Element>
    public let id: Int
    public let isReadOnly: Bool
    public let isReference: Bool
    public var name: String
    
    //--------------------------------------------------------------------------
    // init(count:name:
    @inlinable
    public init(count: Int, name: String) {
        self.hostBuffer = UnsafeMutableBufferPointer.allocate(capacity: count)
        self.id = Context.nextBufferId
        self.isReadOnly = false
        self.isReference = false
        self.name = name
        
        #if DEBUG
        diagnostic("\(createString) \(name)(\(id)) " +
            "\(Element.self)[\(count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(elements:name:
    @inlinable
    public init(copying other: CpuBuffer) {
        self.id = other.id
        self.isReadOnly = other.isReadOnly
        self.isReference = other.isReference
        self.name = other.name
        if isReference {
            hostBuffer = other.hostBuffer
        } else {
            hostBuffer = UnsafeMutableBufferPointer
                .allocate(capacity: other.hostBuffer.count)
            _ = hostBuffer.initialize(from: other.hostBuffer)
        }
    }

    //--------------------------------------------------------------------------
    // init(buffer:name:
    @inlinable
    public init(referenceTo buffer: UnsafeBufferPointer<Element>, name: String) {
        self.hostBuffer = UnsafeMutableBufferPointer(mutating: buffer)
        self.id = Context.nextBufferId
        self.isReadOnly = true
        self.isReference = true
        self.name = name
        
        #if DEBUG
        diagnostic("\(createString) Reference \(name)(\(id)) " +
            "\(Element.self)[\(hostBuffer.count)]", categories: .dataAlloc)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // init(buffer:name:
    @inlinable
    public init(referenceTo buffer: UnsafeMutableBufferPointer<Element>,
                name: String)
    {
        self.hostBuffer = buffer
        self.id = Context.nextBufferId
        self.isReadOnly = false
        self.isReference = true
        self.name = name
        
        #if DEBUG
        diagnostic("\(createString) Reference \(name)(\(id)) " +
            "\(Element.self)[\(hostBuffer.count)]", categories: .dataAlloc)
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
    @inlinable
    deinit {
        if !isReference {
            hostBuffer.deallocate()
            #if DEBUG
            diagnostic("\(releaseString) \(name)(\(id)) ",
                categories: .dataAlloc)
            #endif
        }
    }
    
    @inlinable
    public func element(at offset: Int) -> Element {
        hostBuffer[offset]
    }
    
    @inlinable
    public func setElement(value: Element, at offset: Int) {
        hostBuffer[offset] = value
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable
    public func read(at offset: Int, count: Int) -> UnsafeBufferPointer<Element>
    {
        let start = hostBuffer.baseAddress!.advanced(by: offset)
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable
    public func read(at offset: Int, count: Int, using queue: DeviceQueue)
        -> UnsafeBufferPointer<Element>
    {
        let start = hostBuffer.baseAddress!.advanced(by: offset)
        return UnsafeBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable
    public func readWrite(at offset: Int, count: Int)
        -> UnsafeMutableBufferPointer<Element>
    {
        let start = hostBuffer.baseAddress!.advanced(by: offset)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable
    public func readWrite(at offset: Int, count: Int, willOverwrite: Bool,
                          using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<Element>
    {
        let start = hostBuffer.baseAddress!.advanced(by: offset)
        return UnsafeMutableBufferPointer(start: start, count: count)
    }
}

