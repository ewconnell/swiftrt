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
/// CpuQueueProtocol
public protocol CpuQueueProtocol: DeviceQueue {
    init(id: Int,
         parent logInfo: LogInfo,
         addressing: MemoryAddressing,
         replicationKey: Int,
         deviceId: Int, deviceName: String)
}

//==============================================================================
/// CpuDevice
public struct CpuDevice<Queue>: ComputeDeviceType
    where Queue: CpuQueueProtocol
{
    // properties
    public let id: Int
    public let logInfo: LogInfo
    public let name: String
    public let queues: [Queue]
    
    @inlinable
    public init(parent logInfo: LogInfo, addressing: MemoryAddressing, id: Int)
    {
        let deviceName = "cpu:\(id)"
        let arrayReplicaKey = Current.nextArrayReplicaKey
        self.id = id
        self.name = deviceName
        self.logInfo = logInfo.child(name)
        
        // TODO create 1 queue for each active core
        let queues = [Queue(id: 0, parent: self.logInfo,
                            addressing: addressing,
                            replicationKey: arrayReplicaKey,
                            deviceId: id, deviceName: name)]
        self.queues = queues
    }

    //--------------------------------------------------------------------------
    // createArray
    //    This creates memory on the device
    @inlinable
    public func createArray(byteCount: Int, heapIndex: Int, zero: Bool)
        -> DeviceArray
    {
        CpuDeviceArray(deviceName: name, deviceId: id,
                       addressing: .unified,
                       byteCount: byteCount, zero: zero)
    }
    
    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a device array from a uma buffer.
    @inlinable
    public func createMutableReferenceArray(
        buffer: UnsafeMutableRawBufferPointer) -> DeviceArray
    {
        CpuDeviceArray(deviceName: name, deviceId: id,
                       addressing: .unified, buffer: buffer)
    }
    
    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a device array from a uma buffer.
    @inlinable
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
        -> DeviceArray
    {
        CpuDeviceArray(deviceName: name, deviceId: id,
                       addressing: .unified, buffer: buffer)
    }
}

