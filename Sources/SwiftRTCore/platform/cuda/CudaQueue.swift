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
import Numerics
//import CudaKernels

//==============================================================================
/// CudaQueue
public final class CudaQueue: DeviceQueue, CpuFunctions {
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceId: Int
    public let deviceName: String
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let mode: DeviceQueueMode
    public let name: String
    public let queue: DispatchQueue
    public let useGpu: Bool
    
    public let stream: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle
    @inlinable public var usesCpu: Bool { !useGpu }

    //--------------------------------------------------------------------------
    // initializers
    @inlinable public init(
        parent logInfo: LogInfo,
        deviceId: Int,
        deviceName: String,
        mode: DeviceQueueMode,
        useGpu: Bool
    ) throws {
        self.id = Context.nextQueueId
        self.name = "q\(id)"
        self.logInfo = logInfo.flat(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.memoryType = .discrete
        self.mode = mode
        self.queue = DispatchQueue(label: "\(deviceName)_\(name)")
        self.useGpu = useGpu
        
        // select the specified device
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
        
        // create a queue associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
        stream = cudaStream!
        cudnn = try CudnnHandle(deviceId: deviceId, using: stream)
        cublas = try CublasHandle(deviceId: deviceId, using: stream)
    }
    
    //--------------------------------------------------------------------------
    // select
    public func selectDevice() throws {
        try cudaCheck(status: cudaSetDevice(Int32(self.deviceId)))
    }
    
    //==========================================================================
    /// createActivation
//    public override func createActivation<T>(
//        x: T,
//        y: inout T,
//        mode: ActivationType,
//        nan: NanPropagation,
//        reluCeiling: Double = 0) throws -> ActivationInferring<T>
//        where T: TensorView, T.Element: ScalarElement & FloatingPoint
//    {
//        return try CudaActivationInferring(x: x, y: &y, mode: mode,
//                                           nan: nan, reluCeiling: reluCeiling)
//    }
}
