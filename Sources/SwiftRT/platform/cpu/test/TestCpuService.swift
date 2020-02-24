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

//==============================================================================
/// TestCpuService
/// This is used only for testing. It is an asynchronous cpu version
/// that reports having discreet memory instead of unified to exercise
/// memory management unit tests
public class TestCpuService: PlatformService {
    // properties
    public var deviceBuffers: [Int : DeviceBuffer]
    public let devices: [CpuDevice<CpuQueue>]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [QueueId]

    //--------------------------------------------------------------------------
    @inlinable
    public init() {
        self.deviceBuffers = [Int : DeviceBuffer]()
        self.name = "TestCpuService"
        self.logInfo = LogInfo(logWriter: Platform.log,
                               logLevel: .error,
                               namePath: self.name,
                               nestingLevel: 0)
        self.devices = [
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .unified,  id: 0),
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .discreet, id: 1),
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .discreet, id: 2),
        ]
        
        // select device 0 queue 0 by default
        self.queueStack = []
        self.queueStack = [ensureValidId(0, 0)]
    }
    
    deinit {
        deviceBuffers.values.forEach { $0.deallocate() }
    }
}
