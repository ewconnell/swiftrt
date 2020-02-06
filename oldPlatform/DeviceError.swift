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
/// DeviceError
public enum DeviceError : Error {
    case initializeFailed
    case queueError(idPath: [Int], message: String)
    case timeout(idPath: [Int], message: String)
}

//==============================================================================
/// DeviceErrorHandlerAction
public enum DeviceErrorHandlerAction {
    case doNotPropagate, propagate
}

public typealias DeviceErrorHandler = (Error) -> DeviceErrorHandlerAction

//==============================================================================
/// DeviceErrorHandling
public protocol DeviceErrorHandling: class, _Logging {
    /// user defined handler to override the default
    var deviceErrorHandler: DeviceErrorHandler? { get set }
    /// safe access mutex
    var _errorMutex: Mutex { get }
    /// last error recorded
    var _lastError: Error? { get set }
    
    /// handler that will either call a user handler if defined or propagate
    /// up the device tree
    func handleDevice(error: Error)
}

public extension DeviceErrorHandling {
    /// safe access
    @inlinable
    var lastError: Error? {
        get { _errorMutex.sync { _lastError } }
        set { _errorMutex.sync { _lastError = newValue } }
    }
    
    //--------------------------------------------------------------------------
    /// report(error:event:
    /// sets and propagates a queue error
    /// - Parameter error: the error to report
    @inlinable
    func report(_ error: Error) {
        // set the error state
        lastError = error
        
        // write the error to the log
        writeLog(String(describing: error))
        
        // propagate on app thread
        DispatchQueue.main.async {
            self.handleDevice(error: error)
        }
    }
}

