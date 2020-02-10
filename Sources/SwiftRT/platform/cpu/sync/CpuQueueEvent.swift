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
public final class CpuQueueEvent: QueueEvent, ObjectTracking {
    // properties
    public let trackingId: Int
    public var occurred: Bool
    public var recordedTime: Date?
    public var id: Int { trackingId }

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(options: QueueEventOptions) {
        occurred = true
        trackingId = ObjectTracker.global.nextId
        #if DEBUG
        ObjectTracker.global.register(self)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // deinit
    @inlinable
    deinit {
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    /// wait
    @inlinable
    public func wait() throws { }
}
