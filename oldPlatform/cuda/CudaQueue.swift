//******************************************************************************
// Copyright 2019 Google LLC
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

// @Target(type:"cpu", appliedTo:"CpuSynchronousQueue", protocol: DeviceFunctions)
public final class CudaQueue: LocalDeviceQueue {
    // protocol properties
    public private(set) var trackingId = 0
    public var defaultQueueEventOptions = QueueEventOptions()
    public var device: ComputeDevice { return cudaDevice }
    public let id: Int
    public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()

    /// used to detect accidental queue access by other threads
    private let creatorThread: Thread
    public let cudaDevice: CudaDevice
    public let handle: cudaStream_t
    public let cudnn: CudnnHandle
    public let cublas: CublasHandle

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo, device: CudaDevice,
                name: String, id: Int) throws
    {
        // create a completion event
        cudaDevice = device
        self.logInfo = logInfo
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        let path = logInfo.namePath
        
        // select the specified device
        try cudaDevice.select()
        // create a queue associated with the device
        let flags = UInt32(cudaStreamNonBlocking)
        var cudaStream: cudaStream_t?
        try cudaCheck(status: cudaStreamCreateWithFlags(&cudaStream, flags))
        handle = cudaStream!
        cudnn = try CudnnHandle(deviceId: cudaDevice.id, using: handle,
                                isStatic: true)
        cublas = try CublasHandle(deviceId: cudaDevice.id, using: handle,
                                  isStatic: true)
        trackingId = ObjectTracker.global.register(self, namePath: path,
                                                   isStatic: true)
        
        diagnostic("\(createString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------
    // deinit
    deinit {
        assert(Thread.current === creatorThread,
               "Queue has been captured and is being released by a " +
            "different thread. Probably by a queued function on the queue.")
        
        diagnostic("\(releaseString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: [.queueAlloc])

        do {
            // select the device
            try cudaDevice.select()

            // make sure pending queued commands complete
            // before releasing the queue
            try waitUntilQueueIsComplete()

            // release the queue
            try cudaCheck(status: cudaStreamDestroy(handle))

            // remove from object tracking
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            writeLog(String(describing: error))
        }

        diagnostic("\(releaseString) \(name)", categories: .queueAlloc)
    }

    //--------------------------------------------------------------------------
    // createEvent
    public func createEvent(options: QueueEventOptions) throws -> QueueEvent {
        try cudaDevice.select()
        return try CudaQueueEvent(options: options, timeout: timeout)
    }

//    //--------------------------------------------------------------------------
//    // delay the queue for event testing
//    public func delay(seconds: Double) throws {
//        let clockRate = (device as! CudaDevice).props.clockRate
//        try cudaCheck(status: cudaDelayQueue(seconds, clockRate, handle))
//    }

    //----------------------------------------------------------------------------
    // record
    public func record(event: QueueEvent) throws -> QueueEvent {
        diagnostic("\(recordString) \(name) recording " +
                           "QueueEvent(\(event.trackingId))",
                   categories: .queueSync)
        try cudaDevice.select()
        let event = event as! CudaQueueEvent

        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            event.recordedTime = Date()
        }

        try cudaCheck(status: cudaEventRecord(event.handle, handle))
        return event
    }

    //--------------------------------------------------------------------------
    // wait(for event
    public func wait(for event: QueueEvent) throws {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        diagnostic("\(waitString) \(name) waiting for " +
                           "QueueEvent(\(event.trackingId))",
                   categories: .queueSync)
        try cudaDevice.select()
        let event = event as! CudaQueueEvent
        try cudaCheck(status: cudaStreamWaitEvent(handle, event.handle, 0))
    }

    //--------------------------------------------------------------------------
    // waitUntilQueueIsComplete
    public func waitUntilQueueIsComplete() throws {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        diagnostic("\(blockString) \(name) blocking caller until complete",
                   categories: .queueSync)
        try cudaCheck(status: cudaStreamSynchronize(handle))
    }

    //--------------------------------------------------------------------------
    /// perform indexed copy from source view to result view
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
        fatalError("not implemented yet")
    }
    
    //--------------------------------------------------------------------------
    /// copies from one device array to another
    public func copyAsync(to array: DeviceArray,
                          from otherArray: DeviceArray) throws {
        assert(array is CudaDeviceArray && otherArray is CudaDeviceArray)
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(array.buffer.count == otherArray.buffer.count,
               "buffer sizes don't match")
        try cudaDevice.select()

        // copy
        try cudaCheck(status: cudaMemcpyAsync(
                array.buffer.baseAddress!,
                UnsafeRawPointer(otherArray.buffer.baseAddress!),
                array.buffer.count,
                cudaMemcpyDeviceToDevice, handle))
    }

    //--------------------------------------------------------------------------
    /// copies a host buffer to a device array
    public func copyAsync(to array: DeviceArray,
                          from hostBuffer: UnsafeRawBufferPointer) throws {
        assert(array is CudaDeviceArray)
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        try cudaDevice.select()

        try cudaCheck(status: cudaMemcpyAsync(
                array.buffer.baseAddress!,
                UnsafeRawPointer(hostBuffer.baseAddress!),
                array.buffer.count,
                cudaMemcpyHostToDevice, handle))
    }

    //--------------------------------------------------------------------------
    /// copies a device array to a host buffer
    public func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                          from array: DeviceArray) throws {
        assert(array is CudaDeviceArray)
        assert(hostBuffer.baseAddress != nil)
        assert(hostBuffer.count == array.buffer.count,
               "buffer sizes don't match")
        try cudaDevice.select()

        try cudaCheck(status: cudaMemcpyAsync(
                hostBuffer.baseAddress!,
                UnsafeRawPointer(array.buffer.baseAddress!),
                array.buffer.count,
                cudaMemcpyDeviceToHost, handle))
    }
    
    //--------------------------------------------------------------------------
    /// fills the device array with zeros
    public func zero(array: DeviceArray) throws {
        assert(array is CudaDeviceArray)
        try cudaCheck(
            status: cudaMemsetAsync(array.buffer.baseAddress!, Int32(0),
                                    array.buffer.count, handle))
    }
    
    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the queue by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
            where T: TensorView {
        let delay = TimeInterval(x.shape.elementCount) * timePerElement
        delayQueue(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayQueue(atLeast:
    /// causes the queue to sleep for the specified interval for testing
    public func delayQueue(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
//        queue {
//            Thread.sleep(forTimeInterval: interval)
//        }
    }

    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
//        queue {
//            throw DeviceError.queueError(idPath: [], message: "testError")
//        }
    }
    
    //==========================================================================
    // gemm
    //    Row major matrix multiply
    // A(m x k) * B(k x n) -> C(m x n)
    //
    // http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
    // dgemm("N","N", n, m, k, alpha, B, n, A, k, beta, C, n)
    public func gemm<T>(
        alpha: T.Element = 1,
        transA: TransposeOp, matrixA: T,
        transB: TransposeOp, matrixB: T,
        beta: T.Element = 0,
        matrixC: inout T) throws where
        T: TensorView, T.Element: AnyFloatingPoint
    {
        // make sure the tensors are 2D
        assert(matrixA.rank == 2 && matrixB.rank == 2 && matrixC.rank == 2)
    //    try device.select()
        let m = transA == .noTranspose ? matrixA.extents[0] : matrixA.extents[1]
        let k = transA == .noTranspose ? matrixA.extents[1] : matrixA.extents[0]
        let n = transB == .noTranspose ? matrixB.extents[1] : matrixB.extents[0]
        let rowStrideA = Int32(matrixA.shape.strides[0])
        let rowStrideB = Int32(matrixB.shape.strides[0])
        let rowStrideC = Int32(matrixC.shape.strides[0])
        
        let cudaScalarType = T.Element.scalarType.cuda
        var alpha = alpha
        var beta = beta
        
        // TODO: there are no docs for this, read about cublasGemmAlgo_t
        switch T.Element.scalarType {
        case .real16F:
            try cudaCheck(status: cublasGemmEx(
                cublas.handle,
                transB.cublas, transA.cublas,
                Int32(n), Int32(m), Int32(k),
                &alpha,
                matrixB.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideB,
                matrixA.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideA,
                &beta,
                matrixC.readWrite(using: self).baseAddress!, cudaScalarType,
                rowStrideC,
                ScalarType.real32F.cuda, CUBLAS_GEMM_DFALT))

        case .real32F:
            try cudaCheck(status: cublasGemmEx(
                cublas.handle,
                transB.cublas, transA.cublas,
                Int32(n), Int32(m), Int32(k),
                &alpha,
                matrixB.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideB,
                matrixA.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideA,
                &beta,
                matrixC.readWrite(using: self).baseAddress!, cudaScalarType,
                rowStrideC,
                T.Element.scalarType.cuda, CUBLAS_GEMM_DFALT))

        case .real64F:
            try cudaCheck(status: cublasGemmEx(
                cublas.handle,
                transB.cublas, transA.cublas,
                Int32(n), Int32(m), Int32(k),
                &alpha,
                matrixB.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideB,
                matrixA.readOnly(using: self).baseAddress!, cudaScalarType,
                rowStrideA,
                &beta,
                matrixC.readWrite(using: self).baseAddress!, cudaScalarType,
                rowStrideC,
                T.Element.scalarType.cuda, CUBLAS_GEMM_DFALT))

        default: fatalError("not implemented")
        }
    }

} // CudaQueue

////==============================================================================
//// cudaDataShape(from:)
//// DEFINED IN C Code
//public func cudaDataShape<T>(from tensor: T) -> cudaShape_t where
//    T: TensorView
//{
//    var ptr = UnsafeMutablePointer<cudaShape_t>.allocate(capacity: 1)
//    defer {
//        ptr.deinitialize(count: 1);
//        ptr.deallocate()
//    }
//
//    cudaInitCudaShape(
//            &ptr.pointee,
//            data.scalarType.cuda,
//            data.shape.layout.cudnn,
//            data.extent.count,
//            data.extent,
//            data.strides,
//            data.shape.elementCount)
//
//    return ptr.pointee
//}

//==============================================================================
// CudaReductionContext
public final class CudaReductionContext: ReductionContext {
    // properties
    public let op: ReductionOp
    public let workspace: DeviceArray
    public let workspaceSizeInBytes: Int
    public let reduceTensorDesc: cudnnReduceTensorDescriptor_t
    public let inTensor: TensorDescriptor
    public let outTensor: TensorDescriptor

    //--------------------------------------------------------------------------
    // initializers
    public init(queue: CudaQueue,
                op: ReductionOp,
                scalarType: ScalarType,
                inTensor: TensorDescriptor,
                outTensor: TensorDescriptor) throws {

        self.op = op
        self.inTensor = inTensor
        self.outTensor = outTensor

        var temp: cudnnReduceTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
        reduceTensorDesc = temp!

        let indicesAction = (op == .min || op == .max) ?
                CCuda.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
                CCuda.CUDNN_REDUCE_TENSOR_NO_INDICES

        // adjust intermediate data type if needed
        var reductionDataType: ScalarType
        switch scalarType {
        case .real16F: reductionDataType = .real32F
        default: reductionDataType = scalarType
        }

        try cudaCheck(status: cudnnSetReduceTensorDescriptor(
                reduceTensorDesc,
                op.cudnn,
                reductionDataType.cudnn,
                CCuda.CUDNN_PROPAGATE_NAN,
                indicesAction,
                CCuda.CUDNN_32BIT_INDICES
        ))

        // determine workspace size
        var tempWorkspaceSizeInBytes = 0
        try cudaCheck(status: cudnnGetReductionWorkspaceSize(
                queue.cudnn.handle,
                reduceTensorDesc,
                inTensor.desc,
                outTensor.desc,
                &tempWorkspaceSizeInBytes
        ))
        workspaceSizeInBytes = tempWorkspaceSizeInBytes
        workspace = try queue.device
            .createArray(byteCount: workspaceSizeInBytes,
                         heapIndex: 0,
                         zero: false)
    }
}
