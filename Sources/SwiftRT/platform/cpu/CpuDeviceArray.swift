//******************************************************************************
// Copyright 2019 Google LLC
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
public final class CpuDeviceArray : DeviceArray {
    // properties
    public let buffer: UnsafeMutableRawBufferPointer
    public let deviceId: Int
    public let deviceName: String
    public let isReadOnly: Bool
    public let memoryAddressing: MemoryAddressing
    public let trackingId: Int
    public var version: Int
    
    //--------------------------------------------------------------------------
	/// with count
    @inlinable
    public init(deviceName: String, deviceId: Int, addressing: MemoryAddressing,
                byteCount: Int, zero: Bool)
    {
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.memoryAddressing = addressing
        buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<Double>.alignment)
        if zero {
            buffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
        self.version = 0
        self.isReadOnly = false
        self.trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    /// readOnly uma buffer
    @inlinable
    public init(deviceName: String, deviceId: Int, addressing: MemoryAddressing,
                buffer: UnsafeRawBufferPointer)
    {
        assert(buffer.baseAddress != nil)
        self.isReadOnly = true
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.memoryAddressing = addressing
        let pointer = UnsafeMutableRawPointer(mutating: buffer.baseAddress!)
        self.buffer = UnsafeMutableRawBufferPointer(start: pointer,
                                                    count: buffer.count)
        self.version = 0
        self.trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    /// readWrite uma buffer
    @inlinable
    public init(deviceName: String, deviceId: Int, addressing: MemoryAddressing,
                buffer: UnsafeMutableRawBufferPointer)
    {
        self.isReadOnly = false
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.memoryAddressing = addressing
        self.buffer = buffer
        self.version = 0
        self.trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self)
    }

    @inlinable
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
