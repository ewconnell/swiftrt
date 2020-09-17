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
// CudaEvent
/// An event that blocks all callers until signaled, then lets all waiters
/// through
public class CudaEvent: Logging {
    public let id: Int
    @usableFromInline let event: DispatchSemaphore

    @inlinable public required init(_ tensorId: Int) {
        id = tensorId
        event = DispatchSemaphore(value: 0)
    }
    
    // all write operations must complete before going out of scope
    // a negative id is for the placeholder event during initialization
    // before the first readWrite replaces it.
    @inlinable deinit {
        event.wait()
    }

    @inlinable public func signal() {
        diagnostic(.signaled, "Tensor(\(id)) complete", categories: .queueSync)
        event.signal()
    }

    @inlinable public func wait() {
        diagnostic(.wait, "for Tensor(\(id))", categories: .queueSync)
        event.wait()
        event.signal()
    }
}
