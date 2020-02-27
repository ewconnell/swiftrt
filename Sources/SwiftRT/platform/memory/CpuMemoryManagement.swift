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
public class CpuBuffer: ElementBuffer {
    public let rawBuffer: UnsafeMutableRawBufferPointer
    public let id: Int
    public let isReadOnly: Bool
    public let isReference: Bool
    public let name: String
    
    //--------------------------------------------------------------------------
    // init(type:count:name:
    public init<E>(type: E.Type, count: Int, name: String) {
        let byteCount = count * MemoryLayout<E>.size
        self.rawBuffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<E>.alignment)
        self.id = Platform.nextBufferId
        self.isReadOnly = false
        self.isReference = false
        self.name = name
    }
    
    //--------------------------------------------------------------------------
    // init(buffer:name:
    public init<E>(referenceTo buffer: UnsafeBufferPointer<E>, name: String) {
        let roBuffer = UnsafeRawBufferPointer(buffer)
        let pointer = UnsafeMutableRawPointer(mutating: roBuffer.baseAddress!)
        self.rawBuffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                       count: roBuffer.count)
        self.id = Platform.nextBufferId
        self.isReadOnly = true
        self.isReference = true
        self.name = name
    }
    
    //--------------------------------------------------------------------------
    // init(buffer:name:
    public init<E>(referenceTo buffer: UnsafeMutableBufferPointer<E>,
                   name: String)
    {
        self.rawBuffer = UnsafeMutableRawBufferPointer(buffer)
        self.id = Platform.nextBufferId
        self.isReadOnly = false
        self.isReference = true
        self.name = name
    }
    
    //--------------------------------------------------------------------------
    // deinit
    deinit {
        if !isReference {
            rawBuffer.deallocate()
        }
    }
    
    //--------------------------------------------------------------------------
    // duplicate
    public func duplicate() -> ElementBuffer {
        let source = UnsafeRawBufferPointer(rawBuffer)
        let newBuffer = CpuBuffer(type: UInt8.self,
                                  count: rawBuffer.count,
                                  name: name)
        newBuffer.rawBuffer.copyMemory(from: source)
        return newBuffer
    }
    
    //--------------------------------------------------------------------------
    // read
    public func read<E>(type: E.Type, at offset: Int, count: Int,
                        using queue: DeviceQueue) -> UnsafeBufferPointer<E>
    {
        let elements = rawBuffer.bindMemory(to: E.self)
        return UnsafeBufferPointer(
            start: elements.baseAddress!.advanced(by: offset),
            count: count)
    }
    
    //--------------------------------------------------------------------------
    // readWrite
    public func readWrite<E>(type: E.Type, at offset: Int, count: Int,
                             willOverwrite: Bool, using queue: DeviceQueue)
        -> UnsafeMutableBufferPointer<E>
    {
        let elements = rawBuffer.bindMemory(to: E.self)
        return UnsafeMutableBufferPointer(
            start: elements.baseAddress!.advanced(by: offset),
            count: count)
    }
}

//==============================================================================
/// CpuMemoryManagement
/// Compute services that manage asynchronous discreet devices
/// conform to this protocol
public protocol CpuMemoryManagement: MemoryManagement { }

public extension CpuMemoryManagement where Self: PlatformService
{
    func createBuffer<E>(of type: E.Type, count: Int, name: String)
        -> BufferRef
    {
        BufferRef(CpuBuffer(type: E.self, count: count, name: name))
    }
    
    func createBuffer<E, Shape, Stream>(
        of type: E.Type, block shape: Shape,
        bufferedBlocks: Int, stream: Stream) -> (BufferRef, Int)
        where Shape: ShapeProtocol, Stream: BufferStream
    {
        fatalError()
    }
    
    func cachedBuffer<E>(for element: E) -> BufferRef {
        fatalError()
    }
    
    func createReference<E>(to buffer: UnsafeBufferPointer<E>,
                            name: String) -> BufferRef
    {
        BufferRef(CpuBuffer(referenceTo: buffer, name: name))
    }
    
    func createMutableReference<E>(to buffer: UnsafeMutableBufferPointer<E>,
                                   name: String) -> BufferRef
    {
        BufferRef(CpuBuffer(referenceTo: buffer, name: name))
    }
}

