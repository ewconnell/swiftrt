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
// empty module used to conditionally include
// the CpuAsynchronousService components using #if canImport(module)
import CpuSync

//==============================================================================
/// CpuSynchronousService
public final class CpuSynchronousService:
    ComputeService,
    CpuServiceProtocol,
    LocalComputeService
{
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String

    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach { $0.timeout = timeout }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? cpuSynchronousServiceName
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)

        // add uma synchronouse cpu device
        let device = CpuDevice<CpuSynchronousQueue>(service: self,
                                                    deviceId: 0,
                                                    logInfo: logInfo,
                                                    addressing: .unified,
                                                    timeout: timeout)
        devices.append(device)
    }

    @inlinable
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
}
