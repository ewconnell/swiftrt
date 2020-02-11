//******************************************************************************
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
//

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Platform
/// Manages the scope for the current devices, log, and error handlers
public class Current {
    /// the log output object
    @usableFromInline static var logWriter: Log = Log()
    /// the current compute platform for the thread
    @usableFromInline var platform: ComputePlatform
    /// a platform instance unique id for queue events
    @usableFromInline static var queueEventCounter: Int = 0
    /// platform wide unique value for the `ComputeDevice.arrayReplicaKey`
    @usableFromInline static var arrayReplicaKeyCounter: Int = 0

    //--------------------------------------------------------------------------
    /// the Platform log writing object
    @inlinable public static var log: Log {
        get { logWriter }
        set { logWriter = newValue }
    }
    /// a counter used to uniquely identify queue events for diagnostics
    @inlinable static var nextQueueEventId: Int {
        queueEventCounter += 1
        return queueEventCounter
    }
    /// TODO: probably remove this
    @inlinable static var nextArrayReplicaKey: Int {
        arrayReplicaKeyCounter += 1
        return arrayReplicaKeyCounter
    }

    //--------------------------------------------------------------------------
    // this is a singleton to the initializer is internal
    @usableFromInline
    internal init() {
        platform = Platform<CpuService>()
    }
    /// the current Platform for this thread
    /// NOTE: for performance critical applications, this could be redefined
    /// as a non thread local global static variable that is generic
    /// type specific instead of using the `ComputePlatform` existential
    @inlinable
    public static var platform: ComputePlatform {
        get { threadLocal.platform }
        set { threadLocal.platform = newValue }
    }

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the queues stack
    @usableFromInline
    static var threadLocal: Current {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = Current()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }
    
    //--------------------------------------------------------------------------
    /// thread data key
    @usableFromInline
    static let key: pthread_key_t = {
        var key = pthread_key_t()
        pthread_key_create(&key) {
            #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            let _: AnyObject = Unmanaged.fromOpaque($0).takeRetainedValue()
            #else
            let _: AnyObject = Unmanaged.fromOpaque($0!).takeRetainedValue()
            #endif
        }
        return key
    }()
}
