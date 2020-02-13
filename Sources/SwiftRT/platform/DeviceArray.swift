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
// DeviceArray
//    This represents a device data array
public protocol DeviceArray: ObjectTracking {
    /// a pointer to the memory on the device
    var buffer: UnsafeMutableRawBufferPointer { get }
    /// the device id that this array is associated with
    var deviceId: Int { get }
    /// name used logging
    var deviceName: String { get }
    /// `true` if the array is read only
    var isReadOnly: Bool { get }
    /// specifies the type of associated device memory
    var memoryAddressing: MemoryAddressing { get }
    /// the array edit version number used for replication and synchronization
    var version: Int { get set }
}

public extension DeviceArray {
    @inlinable
    var id: Int { trackingId }
}
