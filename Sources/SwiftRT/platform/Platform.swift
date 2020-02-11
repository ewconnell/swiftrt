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
/// LocalPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public struct Platform<Service>: ComputePlatformType
    where Service: ComputeService
{
    // properties
    public let logInfo: LogInfo
    public let name: String
    public let service: Service
    public var queueStack: [(device: Int, queue: Int)]

    //--------------------------------------------------------------------------
    @inlinable
    public init(log: Log? = nil, name: String? = nil) {
        self.name = name ?? "platform"
        logInfo = LogInfo(logWriter: log ?? Current.log,
                          logLevel: .error,
                          namePath: self.name,
                          nestingLevel: 0)
        
        // create the service
        self.service = Service(parent: logInfo, id: 0)
        
        // selecting device 1 should be the first accelerated device
        // if there is only one device, then the index will wrap to zero
        self.queueStack = []
        self.queueStack = [ensureValidIndexes(1, 0)]
    }
}
