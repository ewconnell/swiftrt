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
public class CpuService: PlatformService {
    // properties
    public let devices: [CpuDevice]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [QueueId]

    //--------------------------------------------------------------------------
    @inlinable
    public init() {
        name = "CpuService"
        logInfo = LogInfo(logWriter: Platform.log, logLevel: .error,
                          namePath: name, nestingLevel: 0)
        devices = [
            CpuDevice(parent: logInfo, memoryType: .unified, id: 0)
        ]
        
        // select device 0 queue 0 by default
        queueStack = []
        queueStack = [ensureValidId(0, 0)]
    }
}
