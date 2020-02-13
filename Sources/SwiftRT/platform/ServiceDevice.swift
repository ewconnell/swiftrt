////******************************************************************************
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

//==============================================================================
/// ServiceDevice
/// a compute device represents a physical service device installed
/// on the platform
public protocol ServiceDevice: Logger {
    /// the id of the device for example dev:0, dev:1, ...
    var id: Int { get }
    /// name used logging
    var name: String { get }
    
    //-------------------------------------
    // device resource functions
    /// creates an array on this device
    /// - Parameter byteCount: the number of bytes to allocate on the device
    /// - Parameter heapIndex: the index of the heap to use
    /// - Parameter zero: `true` to inialize the array to zero
    func createArray(byteCount: Int, heapIndex: Int, zero: Bool) -> DeviceArray
    /// creates a device array from a uma buffer.
    /// - Parameter buffer: a read only byte buffer in the device's
    /// address space
    func createReferenceArray(buffer: UnsafeRawBufferPointer) -> DeviceArray
    /// creates a device array from a uma buffer.
    /// - Parameter buffer: a read write byte buffer in the device's
    /// address space
    func createMutableReferenceArray(buffer: UnsafeMutableRawBufferPointer)
        -> DeviceArray
    /// - Returns: selected queue interface
    func queue(_ index: Int) -> DeviceQueue
}

// version that includes the generic components
public protocol ServiceDeviceType: ServiceDevice {
    associatedtype Queue: DeviceQueue
    /// a collection of device queues for scheduling work
    var queues: [Queue] { get }
}

public extension ServiceDeviceType {
    func queue(_ index: Int) -> DeviceQueue {
        queues[index]
    }
}

