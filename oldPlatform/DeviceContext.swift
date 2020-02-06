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

#if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
import Darwin
#else
import Glibc
#endif

//==============================================================================
/// Executes a closure on the specified queue
/// - Parameter device: the device to set as the single member of the
///  `currentDevices` collection, which is used to execute work
/// - Parameter body: A closure whose operations are to be executed on the
///             specified device
@inlinable
public func using<R>(_ device: ComputeDevice,
                     perform body: () throws -> R) rethrows -> R {
    // sets the default queue and logging info for the current scope
    DeviceContext.local.push(devices: [device])
    defer { DeviceContext.local.popDevices() }
    // execute the body
    return try body()
}

//==============================================================================
/// Executes a closure on the specified queue
/// - Parameter devices: a collection of devices to set as the `currentDevices`
///   collection, which can be used to distribute work
/// - Parameter body: A closure whose operations are to be executed on the
///             specified device
@inlinable
public func using<R>(_ devices: [ComputeDevice],
                     perform body: () throws -> R) rethrows -> R {
    // sets the default queue and logging info for the current scope
    DeviceContext.local.push(devices: devices)
    defer { DeviceContext.local.popDevices() }
    // execute the body
    return try body()
}

//==============================================================================
/// DeviceContext
/// Manages the scope for the current devices, log, and error handlers
public class DeviceContext {
    /// stack of current device collections used to execute/distribute work
    public var devicesStack: [[ComputeDevice]]
    /// specifies whether operators in the current scope are
    /// evaluated for training or inferring
    public var evaluationMode: EvaluationMode
    /// a convenience property. `true` if the context is inferring
    @inlinable
    public static var isInferring: Bool {
        DeviceContext.local.evaluationMode == .inferring
    }

    /// a convenience property. `true` if the context is training
    @inlinable
    public static var isTraining: Bool {
        DeviceContext.local.evaluationMode == .training
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

    //--------------------------------------------------------------------------
    /// returns the thread local instance of the queues stack
    @inlinable
    public static var local: DeviceContext {
        // try to get an existing state
        if let state = pthread_getspecific(key) {
            return Unmanaged.fromOpaque(state).takeUnretainedValue()
        } else {
            // create and return new state
            let state = DeviceContext()
            pthread_setspecific(key, Unmanaged.passRetained(state).toOpaque())
            return state
        }
    }

    //--------------------------------------------------------------------------
    /// current
    @inlinable
    public static var current: [ComputeDevice] {
        DeviceContext.local.devicesStack.last!
    }

    //--------------------------------------------------------------------------
    /// currentQueue
    // TODO: temporary scheme
    @inlinable
    public static var currentQueue: DeviceQueue {
        DeviceContext.local.devicesStack.last![0].queues[0]
    }
    
    //--------------------------------------------------------------------------
    /// hostQueue
    @inlinable
    public static var hostQueue: DeviceQueue {
        assert(Platform.cpu.memory.addressing == .unified)
        
        let current = DeviceContext.current[0]
        return current.memory.addressing == .unified ?
            current.queueueues[0] : Platform.cpu.queues[0]
    }

    //--------------------------------------------------------------------------
    /// logInfo
    // `last` is always valid because there will always be
    // the platform default queue and logInfo
    @inlinable
    public var logInfo: LogInfo { devicesStack.last![0].logInfo }

    //--------------------------------------------------------------------------
    @inlinable
    public static var lastError: Error? {
        get { DeviceContext.currentQueue.device.lastError }
        set { DeviceContext.currentQueue.device.lastError = newValue }
    }
    
    //--------------------------------------------------------------------------
    /// report(error:event:
    /// sets and propagates an error on the current device
    /// - Parameter error: the error to report
    @inlinable
    public static func report(_ error: Error) {
        DeviceContext.currentQueue.device.report(error)
    }

    //--------------------------------------------------------------------------
    // initializers
    @usableFromInline
    init() {
        devicesStack = [[Platform.local.defaultDevice]]
        evaluationMode = .inferring
    }

    //--------------------------------------------------------------------------
    /// push(devices:
    /// pushes the specified device collection onto a queue stack which makes
    /// it the current queue used by operator functions
    @inlinable
    func push(devices: [ComputeDevice]) {
        devicesStack.append(devices)
    }

    //--------------------------------------------------------------------------
    /// popDevices
    /// restores the previous current devices collection
    @inlinable
    func popDevices() {
        assert(devicesStack.count > 1)
        _ = devicesStack.popLast()
    }
}
