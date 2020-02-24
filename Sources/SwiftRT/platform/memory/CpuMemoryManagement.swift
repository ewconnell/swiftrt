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
/// Used internally to manage the state of a collection of device buffers
public struct CpuBuffer {
    /// the number of bytes in the buffer
    public let byteCount: Int
    /// a dictionary of device memory replicas allocated on each device
    /// - Parameter key: the device index
    /// - Returns: the associated device memory object
    public var memory: [Int : DeviceMemory]
    
    /// `true` if the buffer is not mutable, such as in the case of a readOnly
    /// reference buffer.
    public let isReadOnly: Bool
    
    /// the `id` of the last queue that obtained write access
    public var lastMutatingQueue: QueueId
    
    /// the buffer name used in diagnostic messages
    public let name: String
    
    /// the index of the device holding the master version
    public var masterDevice: Int
    
    /// the masterVersion is incremented each time write access is taken.
    /// All device buffers will stay in sync with this version, copying as
    /// necessary.
    public var masterVersion: Int
    
    /// helper to return `Element` sized count
    @inlinable
    public func count<Element>(of type: Element.Type) -> Int {
        byteCount * MemoryLayout<Element>.size
    }
    
    //--------------------------------------------------------------------------
    /// initializer
    @inlinable
    public init(byteCount: Int, name: String, isReadOnly: Bool = false) {
        self.byteCount = byteCount
        self.memory = [Int : DeviceMemory]()
        self.isReadOnly = isReadOnly
        self.lastMutatingQueue = QueueId(0, 0)
        self.masterDevice = 0
        self.masterVersion = 0
        self.name = name
    }
    
    //--------------------------------------------------------------------------
    /// `deallocate`
    /// releases device memory associated with this buffer descriptor
    /// - Parameter device: the device to release memory from. `nil` will
    /// release all associated device memory for this buffer.
    @inlinable
    public func deallocate(device: Int? = nil) {
        if let device = device {
            memory[device]!.deallocate()
        } else {
            memory.values.forEach { $0.deallocate() }
        }
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


}

