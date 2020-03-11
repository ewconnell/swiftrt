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
import CCuda
//import CudaKernels

//==============================================================================
/// CudaQueue
public final class CudaQueue: DeviceQueue {
    public let useGpu: Bool
    
    public let handle: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                deviceId: Int, deviceName: String, useGpu: Bool) throws
    {
        self.useGpu = useGpu
        
        // select the specified device
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
        
        // create a queue associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
        handle = cudaStream!
        cudnn = try CudnnHandle(deviceId: deviceId, using: handle, isStatic: true)
        cublas = try CublasHandle(deviceId: deviceId, using: handle,
                                  isStatic: true)

        super.init(id: id, parent: logInfo, deviceId: deviceId,
                   deviceName: deviceName,
                   memoryType: useGpu ? .discreet : .unified)
    }
    
    //--------------------------------------------------------------------------
    // select
    public func selectDevice() throws {
        try cudaCheck(status: cudaSetDevice(Int32(self.deviceId)))
    }
}

