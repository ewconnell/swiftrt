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
/// CpuMemoryManager
public struct CpuMemoryManager: MemoryManagement {
    public var deviceBuffers: [[Int : BufferDescription]]
    public var masterVersion: [Int : Int]
    
    @inlinable
    public init() {
        deviceBuffers = [[Int : BufferDescription]]()
        masterVersion = [Int : Int]()
    }
}


//==============================================================================
// stubs
public extension MemoryManagement {
    func bufferName(_ id: BufferId) -> String { fatalError() }
    func cachedBuffer<Element>(for element: Element) -> BufferId { fatalError() }
    func createBuffer<T>(of type: T.Type, count: Int, name: String) -> BufferId { fatalError() }
    func createBuffer<Shape, Stream>(block shape: Shape, bufferedBlocks: Int, stream: Stream) -> (BufferId, Int)
        where Shape: ShapeProtocol, Stream: BufferStream { fatalError() }
    func createReference<Element>(to applicationBuffer: UnsafeBufferPointer<Element>) -> BufferId { fatalError() }
    func createMutableReference<Element>(to applicationBuffer: UnsafeMutableBufferPointer<Element>) -> BufferId  { fatalError() }
    func duplicate(_ buffer: BufferId, using queue: QueueId) -> BufferId  { fatalError() }
    func release(_ buffer: BufferId)  { fatalError() }
    func read<T>(_ buffer: BufferId, of type: T.Type, at offset: Int, count: Int, using queue: DeviceQueue) -> UnsafeBufferPointer<T> { fatalError() }
    func readWrite<T>(_ buffer: BufferId, of type: T.Type, at offset: Int, count: Int, willOverwrite: Bool, using queue: DeviceQueue) -> UnsafeMutableBufferPointer<T> { fatalError() }
}
