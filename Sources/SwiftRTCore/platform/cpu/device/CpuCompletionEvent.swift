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
// CpuCompletionEvent
/// a queue event behaves like a barrier. The first caller to wait takes
/// the wait semaphore
public class CpuCompletionEvent: CompletionEvent {
    @usableFromInline var occurred: Bool
    @usableFromInline let mutex: DispatchSemaphore
    @usableFromInline let event: DispatchSemaphore

    /// the id of the event for diagnostics
    @inlinable public var id: Int { 0 }

    @inlinable public init() {
        occurred = false
        mutex = DispatchSemaphore(value: 1)
        event = DispatchSemaphore(value: 0)
    }
    
    @inlinable deinit {
        // all write operations must complete before going out of scope
        wait()
    }

    @inlinable public func signal() {
        event.signal()
    }

    @inlinable public func wait() {
        mutex.wait()
        defer { mutex.signal() }
        guard !occurred else { return }
        event.wait()
        occurred = true
    }
}
