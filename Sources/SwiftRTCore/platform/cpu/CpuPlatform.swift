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
/// CpuPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class CpuPlatform: Platform {
    // properties
    public var devices: [CpuDevice]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [CpuQueue]
    public let syncQueue: CpuQueue

    //--------------------------------------------------------------------------
    @inlinable public init() {
        name = "\(Self.self)"
        logInfo = LogInfo(logWriter: Context.log, logLevel: .error,
                          namePath: name, nestingLevel: 0)
        
        // make the first queue the sync queue so diagnostics are
        // easier to read
        let syncQueueId = Context.nextQueueId
        
        // create the device and default number of async queues
        let device = CpuDevice(id: 0, parent: logInfo, memoryType: .unified)
        devices = [device]

        // create the application thread data interchange queue
        syncQueue = CpuQueue(queueId: syncQueueId,
                             parent: device.logInfo,
                             deviceId: device.id,
                             deviceName: device.name,
                             memoryType: .unified,
                             mode: .sync)
        
        // if the number of requested async queues is 0, then make the
        // syncQueue the default
        queueStack = device.queues.count == 0 ? [syncQueue] : [device.queues[0]]
    }
}
