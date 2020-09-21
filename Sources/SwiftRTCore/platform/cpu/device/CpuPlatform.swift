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
import Foundation

//==============================================================================
/// CpuPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class CpuPlatform: Platform {
    // types
    public typealias Storage = CpuStorage
    
    // properties
    public var discreteMemoryDeviceId: Int { 1 }
    public var devices: [CpuDevice]
    public let name: String
    public var queueStack: [CpuQueue]
    public let appThreadQueue: CpuQueue
    public static let defaultAcceleratorQueueCount: Int = 0
    public static var defaultCpuQueueCount =
        ProcessInfo.processInfo.activeProcessorCount

    //--------------------------------------------------------------------------
    @inlinable public init() {
        name = "\(Self.self)"
        
        // create the device and default number of async queues
        let device = CpuDevice(index: 0, memoryType: .unified,
                               queueCount: Context.cpuQueueCount)
        
        // create a discrete memory unit test device
        let test = CpuDevice(index: 1, memoryType: .discrete, queueCount: 2)
        devices = [device, test]

        // create the application thread data interchange queue
        appThreadQueue = CpuQueue(deviceIndex: 0,
                             name: "appThread",
                             queueMode: .sync,
                             memoryType: .unified)

        // make the app thread queue current by default
        queueStack = [appThreadQueue]

        diagnostic(.device, "create  sync queue: appThread",
                   categories: .queueAlloc)
    }
}
