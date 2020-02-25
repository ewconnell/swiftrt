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
/// CpuBuffer
/// Used to manage a host memory buffer
public struct CpuBuffer {
    /// host memory buffer pointer
    public let buffer: UnsafeMutableRawBufferPointer
    /// function to free the memory
    public let deallocate: () -> Void
    /// `true` if the buffer is not mutable, such as in the case of a readOnly
    /// reference buffer.
    public let isReadOnly: Bool
    /// the buffer name used in diagnostic messages
    public let name: String
    /// helper to return `Element` sized count
    @inlinable
    public func count<Element>(of type: Element.Type) -> Int {
        buffer.count * MemoryLayout<Element>.size
    }
    
    //--------------------------------------------------------------------------
    /// initializer
    @inlinable
    public init(_ buffer: UnsafeMutableRawBufferPointer,
                _ name: String, isReadOnly: Bool = false,
                _ deallocate: @escaping () -> Void)
    {
        self.buffer = buffer
        self.isReadOnly = isReadOnly
        self.name = name
        self.deallocate = deallocate
    }
}

//==============================================================================
/// CpuMemoryManagement
/// Compute services that manage asynchronous discreet devices
/// conform to this protocol
public protocol CpuMemoryManagement: MemoryManagement {
    /// a dictionary of device buffer entries indexed by the device
    /// number, and keyed by the id returned from `createBuffer`.
    /// By convention device 0 will always be a unified memory device with
    /// the application.
    var deviceBuffers: [Int : CpuBuffer] { get set }
}

public extension CpuMemoryManagement where Self: PlatformService {
    //--------------------------------------------------------------------------
    // bufferName
    @inlinable
    func bufferName(_ ref: BufferRef) -> String {
        deviceBuffers[ref.id]!.name
    }
    
    //--------------------------------------------------------------------------
    // createBuffer
    @inlinable
    func createBuffer<Element>(of type: Element.Type, count: Int,
                               name: String) -> BufferRef
    {
        let ref = self.nextBufferRef
        let byteCount = count * MemoryLayout<Element>.size
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<Double>.alignment)
        deviceBuffers[ref.id] = CpuBuffer(buffer, name, { buffer.deallocate() })
        return ref
    }
    
    //--------------------------------------------------------------------------
    // createBuffer
    @inlinable
    func createBuffer<Shape, Stream>(block shape: Shape, bufferedBlocks: Int,
                                     stream: Stream) -> (BufferRef, Int)
        where Shape : ShapeProtocol, Stream : BufferStream
    {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // cachedBuffer
    @inlinable
    func cachedBuffer<Element>(for element: Element) -> BufferRef {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // createReference
    @inlinable
    func createReference<Element>(to buffer: UnsafeBufferPointer<Element>,
                                  name: String) -> BufferRef
    {
        // get a reference id
        let ref = self.nextBufferRef
        
        // create a device buffer entry for the id
        let roBuffer = UnsafeRawBufferPointer(buffer)
        let pointer = UnsafeMutableRawPointer(mutating: roBuffer.baseAddress!)
        let rawBuffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                      count: roBuffer.count)
        deviceBuffers[ref.id] = CpuBuffer(rawBuffer, name, isReadOnly: true, {})
        return ref
    }
    
    //--------------------------------------------------------------------------
    // createMutableReference
    @inlinable
    func createMutableReference<Element>(
        to buffer: UnsafeMutableBufferPointer<Element>,
        name: String) -> BufferRef
    {
        // get a reference id
        let ref = self.nextBufferRef
        
        // create a device buffer entry for the id
        let rawBuffer = UnsafeMutableRawBufferPointer(buffer)
        deviceBuffers[ref.id] = CpuBuffer(rawBuffer, name, {})
        return ref
    }
    
    //--------------------------------------------------------------------------
    // duplicate
    @inlinable
    func duplicate(_ ref: BufferRef, using queue: QueueId) -> BufferRef {
        let source = deviceBuffers[ref.id]!
        let sourceBuffer = UnsafeRawBufferPointer(source.buffer)
        let newRef = createBuffer(of: UInt8.self, count: sourceBuffer.count,
                                  name: source.name)
        deviceBuffers[newRef.id]!.buffer.copyMemory(from: sourceBuffer)
        return newRef
    }
    
    //--------------------------------------------------------------------------
    // release
    @inlinable
    func release(_ ref: BufferRef) {
        deviceBuffers[ref.id]!.buffer.deallocate()
        deviceBuffers.removeValue(forKey: ref.id)
    }
    
    //--------------------------------------------------------------------------
    // read
    @inlinable
    func read<Element>(_ ref: BufferRef, of type: Element.Type, at offset: Int,
                       count: Int, using queueId: QueueId)
        -> UnsafeBufferPointer<Element>
    {
        let pointer = deviceBuffers[ref.id]!.buffer.bindMemory(to: Element.self)
        return UnsafeBufferPointer(
            start: pointer.baseAddress!.advanced(by: offset),
            count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    @inlinable
    func readWrite<Element>(_ ref: BufferRef, of type: Element.Type,
                            at offset: Int, count: Int, willOverwrite: Bool,
                            using queueId: QueueId)
        -> UnsafeMutableBufferPointer<Element>
    {
        let pointer = deviceBuffers[ref.id]!.buffer.bindMemory(to: Element.self)
        return UnsafeMutableBufferPointer(
            start: pointer.baseAddress!.advanced(by: offset),
            count: count)
    }
}

