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
/// CpuService
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class CpuService: PlatformService, CpuMemoryManagement {
    // properties
    public var deviceBuffers: [Int : CpuBuffer]
    public let devices: [CpuDevice<CpuQueue>]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [QueueId]

    //--------------------------------------------------------------------------
    @inlinable
    public init() {
        self.deviceBuffers = [Int : CpuBuffer]()
        self.name = "CpuService"
        self.logInfo = LogInfo(logWriter: Platform.log,
                               logLevel: .error,
                               namePath: self.name,
                               nestingLevel: 0)
        self.devices = [
            CpuDevice<CpuQueue>(parent: logInfo, memoryType: .unified, id: 0)
        ]
        
        // select device 0 queue 0 by default
        self.queueStack = []
        self.queueStack = [ensureValidId(0, 0)]
    }
    
    deinit {
        deviceBuffers.values.forEach { $0.deallocate() }
    }

    public func bufferName(_ ref: BufferRef) -> String {
        fatalError()
    }
    
    public func createBuffer<Element>(of type: Element.Type, count: Int, name: String) -> BufferRef {
        fatalError()
    }
    
    public func createBuffer<Shape, Stream>(block shape: Shape, bufferedBlocks: Int, stream: Stream) -> (BufferRef, Int) where Shape : ShapeProtocol, Stream : BufferStream {
        fatalError()
    }
    
    public func cachedBuffer<Element>(for element: Element) -> BufferRef {
        fatalError()
    }
    
    public func createReference<Element>(to buffer: UnsafeBufferPointer<Element>, name: String) -> BufferRef {
        fatalError()
    }
    
    public func createMutableReference<Element>(to buffer: UnsafeMutableBufferPointer<Element>, name: String) -> BufferRef {
        fatalError()
    }
    
    public func duplicate(_ other: BufferRef, using queue: QueueId) -> BufferRef {
        fatalError()
    }
    
    public func release(_ ref: BufferRef) {
        fatalError()
    }
    
    public func read<Element>(_ ref: BufferRef, of type: Element.Type, at offset: Int, count: Int, using queueId: QueueId) -> UnsafeBufferPointer<Element> {
        fatalError()
    }
    
    public func readWrite<Element>(_ ref: BufferRef, of type: Element.Type, at offset: Int, count: Int, willOverwrite: Bool, using queueId: QueueId) -> UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
}
