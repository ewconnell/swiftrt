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
import Foundation

//==============================================================================
// CpuQueueEvent
/// a queue event behaves like a barrier. The first caller to wait takes
/// the wait semaphore
public class CpuQueueEvent: QueueEvent {
    // properties
    public var occurred: Bool
    public var recordedTime: Date?
    public let id: Int
    public let semaphore: DispatchSemaphore

    // initializers
    @inlinable public init(options: QueueEventOptions) {
        id = Context.nextQueueEventId
        occurred = false
        semaphore = DispatchSemaphore(value: 0)
    }

    // signals that the event has occurred
    public func signal() {
        #if DEBUG
        Context.currentQueue.diagnostic(
            "\(signaledString) QueueEvent(\(id))", categories: .queueSync)
        #endif
        occurred = true
        semaphore.signal()
    }
    
    // wait
    // a synchronous queue event is a noop since all ops are completed
    // at the time they are queued
    @inlinable public func wait() {
        semaphore.wait()
    }
}
