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
import Foundation

// assert messages
let _messageQueueThreadViolation =
"a queue can only be accessed by the thread that created it"

//==============================================================================
/// DeviceQueue
/// A device queue is an asynchronous sequential list of commands to be
/// executed on the associated device.
public protocol DeviceQueue: Logger, DeviceFunctions {
    /// a key to lookup a DeviceArray replica associated with this device
    var arrayReplicaKey: Int { get }
    /// the thread that created this queue. Used to detect accidental access
    var creatorThread: Thread { get }
    /// options to use when creating queue events
    var defaultQueueEventOptions: QueueEventOptions { get }
    /// the device id that this queue is associated with
    var deviceId: Int { get }
    /// the id of the device for example queue:0, queue:1, ...
    var id: Int { get }
    /// name used logging
    var deviceName: String { get }
    /// specifies the type of associated device memory
    var memoryAddressing: MemoryAddressing { get }
    /// name used logging
    var name: String { get }

    //--------------------------------------------------------------------------
    // synchronization functions
    /// creates a QueueEvent
    func createEvent(options: QueueEventOptions) -> QueueEvent
    /// queues a queue event op. When executed the event is signaled
    @discardableResult
    func record(event: QueueEvent) -> QueueEvent
    /// records an op on the queue that will perform a queue blocking wait
    /// when it is processed
    func wait(for event: QueueEvent)
    /// blocks the calling thread until the queue queue has completed all work
    func waitUntilQueueIsComplete()
    
    //--------------------------------------------------------------------------
    /// copy
    /// performs an indexed copy from view to result
    func copy<T>(from view: T, to result: inout T) where T: TensorView
    /// asynchronously copies the contents of another device array
    func copyAsync(to array: DeviceArray, from otherArray: DeviceArray)
    /// asynchronously copies the contents of an app memory buffer
    func copyAsync(to array: DeviceArray,
                   from hostBuffer: UnsafeRawBufferPointer)
    /// copies the contents to an app memory buffer asynchronously
    func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                   from array: DeviceArray)
    /// clears the array to zero
    func zero(array: DeviceArray)
}

public extension DeviceQueue {
    @inlinable
    func createEvent() -> QueueEvent {
        createEvent(options: defaultQueueEventOptions)
    }

    @inlinable
    var device: ServiceDevice { Current.platform.device(deviceId) }
}

//==============================================================================
/// MemoryAddressing
public enum MemoryAddressing {
    case unified, discreet
}
