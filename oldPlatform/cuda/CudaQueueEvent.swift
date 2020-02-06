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
import CCuda

public final class CudaQueueEvent : QueueEvent {
    // properties
    public private(set) var trackingId = 0
    public var recordedTime: Date?
    
    public let options: QueueEventOptions
    private let timeout: TimeInterval?
    public let handle: cudaEvent_t
    private let barrier = Mutex()
    private let semaphore = DispatchSemaphore(value: 0)
    
    //--------------------------------------------------------------------------
    // occurred
    public var occurred: Bool {
        return cudaEventQuery(handle) == cudaSuccess
    }
    
    //--------------------------------------------------------------------------
    // initializers
    public init(options: QueueEventOptions, timeout: TimeInterval?) throws {
        self.options = options
        self.timeout = timeout
        
        // the default is non host blocking, non timing, non inter-process
        var temp: cudaEvent_t?
        try cudaCheck(status: cudaEventCreateWithFlags(
            &temp, UInt32(cudaEventDisableTiming)))
        handle = temp!
        
        #if DEBUG
        trackingId = ObjectTracker.global.register(self)
        #endif
    }
    
    //--------------------------------------------------------------------------
    // deinit
    deinit {
        _ = cudaEventDestroy(handle)
        
        #if DEBUG
        ObjectTracker.global.remove(trackingId: trackingId)
        #endif
    }
    
    //--------------------------------------------------------------------------
    /// wait
    public func wait() throws {
        try barrier.sync {
            guard !occurred else { return }
            if let timeout = timeout {
                var elapsed = 0.0;
                // TODO: is 1 millisecond reasonable?
                let pollingInterval = 0.001
                while (cudaEventQuery(handle) != cudaSuccess) {
                    Thread.sleep(forTimeInterval: pollingInterval)
                    elapsed += pollingInterval;
                    if (elapsed >= timeout) {
                        throw QueueEventError.timedOut
                    }
                }
            } else {
                // wait indefinitely
                try cudaCheck(status: cudaEventSynchronize(handle))
            }
        }
    }
}
