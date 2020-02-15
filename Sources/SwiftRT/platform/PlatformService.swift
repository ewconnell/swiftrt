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

//==============================================================================
/// PlatformService
/// a compute service represents a category of installed devices on the
/// platform, such as (cpu, cuda, tpu, ...)
public protocol PlatformService: ServiceMemoryManagement, Logger {
    /// service id used for logging, usually zero
    var id: Int { get }
    /// name used logging
    var name: String { get }
    
    //--------------------------------------------------------------------------
    // initializers
    init(parent parentLogInfo: LogInfo, id: Int)
}

//==============================================================================
/// QueueId
/// a unique service device queue identifier that is used to index
/// through the service device tree for directing workflow
public struct QueueId {
    public let service: Int
    public let device: Int
    public let queue: Int
    public init(_ service: Int, _ device: Int, _ queue: Int) {
        self.service = service
        self.device = device
        self.queue = queue
    }
}

//==============================================================================
/// PlatformServiceType
public protocol PlatformServiceType: PlatformService {
    // types
    associatedtype Device: ServiceDeviceType
    
    /// a collection of available compute devices
    var devices: [Device] { get }
}

