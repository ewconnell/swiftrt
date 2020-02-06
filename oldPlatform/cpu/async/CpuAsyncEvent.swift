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
public final class CpuAsyncEvent : QueueEvent {
    // properties
    public var trackingId: Int
    public var occurred: Bool
    public var recordedTime: Date?

    public let options: QueueEventOptions
    public let timeout: TimeInterval?
    public let barrier: Mutex
    public let semaphore: DispatchSemaphore

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(options: QueueEventOptions, timeout: TimeInterval?) {
        self.options = options
        self.timeout = timeout
        self.barrier = Mutex()
        self.occurred = false
        self.semaphore = DispatchSemaphore(value: 0)
        trackingId = 0
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // deinit
    @inlinable
    deinit {
        // signal if anyone was waiting
        signal()
        
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    /// signal
    /// signals that the event has occurred
    @inlinable
    public func signal() {
        semaphore.signal()
    }
    
    //--------------------------------------------------------------------------
    /// wait
    /// the first thread goes through the barrier.sync and waits on the
    /// semaphore. When it is signaled `occurred` is set to `true` and all
    /// future threads will pass through without waiting
    @inlinable
    public func wait() throws {
        try barrier.sync {
            guard !occurred else { return }
            if let timeout = self.timeout, timeout > 0 {
                if semaphore.wait(timeout: .now() + timeout) == .timedOut {
                    throw QueueEventError.timedOut
                }
            } else {
                semaphore.wait()
            }
            occurred = true
        }
    }
}
