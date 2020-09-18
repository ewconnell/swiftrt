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
import SwiftRTCuda

//==============================================================================
// CudaEvent
/// An event that blocks all until signaled, then lets all through
public class CudaEvent: Logging {
	public var handle: cudaEvent_t

    #if DEBUG
    public let id = Context.nextEventId
    #else
    public let id = 0
    #endif

    @inlinable public required init(options: QueueEventOptions = []) {
		// the default is a non host blocking, non timing, non inter process event
		var flags: Int32 = cudaEventDisableTiming
		if !options.contains(.timing)      { flags &= ~cudaEventDisableTiming }
		if options.contains(.interprocess) { flags |= cudaEventInterprocess |
			                                            cudaEventDisableTiming }
		// if options.contains(.hostSync)     { flags |= cudaEventBlockingSync }

		var temp: cudaEvent_t?
		cudaCheck(cudaEventCreateWithFlags(&temp, UInt32(flags)))
		handle = temp!
    }
    
    // event must be signaled before going out of scope to ensure 
    @inlinable deinit {
		_ = cudaEventDestroy(handle)
    }

    @inlinable public func signal() {
        fatalError()
    }

    @inlinable public func wait() {
        fatalError()
    }
}
