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
/// CpuService
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public struct CpuService: PlatformService {
    // properties
    public let devices: [CpuDevice<CpuQueue>]
    public var deviceBuffers: [[Int : BufferDescription]]
    public let id: Int
    public let logInfo: LogInfo
    public var masterVersion: [Int : Int]
    public let name: String

    //--------------------------------------------------------------------------
    @inlinable
    public init(parent parentLogInfo: LogInfo, id: Int) {
        self.name = "CpuService"
        self.logInfo = parentLogInfo.child(name)
        self.id = id
        self.deviceBuffers = [[Int : BufferDescription]]()
        self.masterVersion = [Int : Int]()

        self.devices = [
            CpuDevice<CpuQueue>(parent: logInfo, addressing: .unified, id: 0)
        ]
    }
}
