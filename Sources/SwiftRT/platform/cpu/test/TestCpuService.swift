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
public struct TestCpuService: ComputeService {
    // properties
    public let devices: [CpuDevice<CpuQueue>]
    public let id: Int
    public let logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    @inlinable
    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.name = "TestCpuService"
        self.logInfo = parentLogInfo.child(name)
        self.id = id
        self.devices = [
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .unified,  id: 0),
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .discreet, id: 1),
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .discreet, id: 2),
        ]
    }
}
