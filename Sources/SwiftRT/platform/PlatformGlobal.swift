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
// setGlobal(platform:
// NOTE: do this in your app if the source is part of the app

// let globalPlatform = Platform<XyzService>()
// You can define APPCOLOCATED in the build, or just delete this file

//==============================================================================
/// setGlobal(platform:
/// this is used to set the framework global variable for static functions
/// and free floating objects to access the platform
#if !APPCOLOCATED
@inlinable
public func setGlobal<T>(platform: T) -> T where T: ComputePlatform {
    globalPlatform = platform
    return platform
}

/// This is an existential, so it is slower than if the
public var globalPlatform: ComputePlatform = Platform<CpuService>()
#endif

//==============================================================================
/// PlatformStatic
/// Platform is a generic type which are not allowed to contain stored
/// static properties, so they are stored here.
public struct PlatformStatic {
    /// a platform instance unique id for queue events
    @usableFromInline
    static var queueEventCounter: Int = 0
   
    @inlinable
    public static var nextQueueEventId: Int {
        queueEventCounter += 1
        return queueEventCounter
    }
}

//==============================================================================
/// PlatformContext
/// Manages the scope for the current devices, log, and error handlers
public class PlatformContext {
    public var platform: ComputePlatform = Platform<CpuService>()
    
    @usableFromInline
    internal init() { }
    
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

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the queues stack
    @inlinable
    public static var current: PlatformContext {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = PlatformContext()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

}
