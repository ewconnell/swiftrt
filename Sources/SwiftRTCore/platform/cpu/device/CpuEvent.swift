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
// CpuEvent
/// An event that blocks all until signaled, then lets all through
public class CpuEvent: Logging {
    @usableFromInline let event: DispatchSemaphore

    #if DEBUG
    public let id = Context.nextEventId
    #else
    public let id = 0
    #endif

    @inlinable public required init(options: QueueEventOptions = []) {
        event = DispatchSemaphore(value: 0)
    }
    
    // event must be signaled before going out of scope to ensure 
    @inlinable deinit {
        event.wait()
    }

    @inlinable public func signal() {
        event.signal()
    }

    @inlinable public func wait() {
        event.wait()
        event.signal()
    }
}
