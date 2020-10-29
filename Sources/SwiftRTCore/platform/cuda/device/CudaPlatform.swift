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
import SwiftRTCuda

//==============================================================================
/// CudaPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class CudaPlatform: ComputePlatform {
    // types
    public typealias Storage = DiscreteStorage
    
        // shared
    public static var cpuQueueCount = 0
    public static let discreteMemoryDeviceId = 1
    public static let eventId = AtomicCounter()
    public static let local = CudaPlatform()
    public static let mainThread = pthread_self()
    public static let objectId = AtomicCounter()
    public static let queueId = AtomicCounter()
    public static let startTime = Date()
    public static var lastRandomSeed: RandomSeed = generateRandomSeed()

    public static var acceleratorQueueCount: Int = 2 {
        didSet {
            precondition(acceleratorQueueCount > 0, 
                "there must be at least 1 accelerator queue")
        }
    }

    //-------------------------------------
    // for synchrnous execution and syncing with the app thread
    public static let syncQueue =
        CudaQueue(deviceIndex: 0, name: "appThread", 
                  queueMode: .sync, useGpu: false)

    // properties
    public var devices: [CudaDevice]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [CudaQueue]

    //-------------------------------------
    // HACK for AD
    /// a storage buffer with a single zero value which is shared
    /// every time Element.zero is obtained by AD.
    // used to minimize AD overhead. AD needs to fix this problem.
    public static let zeroStorage: Storage = {
        Storage(storedElement: Int64(0), name: "Zero")
    }()

    //--------------------------------------------------------------------------
    // initializer
    @inlinable public init() {
        //----------------------------
        // CudaDevice is overloaded to avoid using Swift existentials
        // to support both cpu and gpu operations.
        // Device 0 is the cpu
        let cpuDevice = CudaDevice(index: 0)
        devices = [cpuDevice]

        // if the cpu queue count is 0 then at least add in
        // the SyncQueue so there is something to work with
        if devices[0].queues.count == 0 {
            devices[0].queues.append(Self.syncQueue)
        }            

        //----------------------------
        // query cuda to get number of installed devices
        queueStack = []
        var gpuDeviceCount: CInt = 0
        cudaCheck(cudaGetDeviceCount(&gpuDeviceCount))

        // add device for each reported gpu
        for i in 1..<Int(gpuDeviceCount + 1) {
            devices.append(CudaDevice(index: i))
        }

        //----------------------------
        // select first gpu queue 0 as default
        name = "\(Self.self)"
        logInfo = LogInfo(logWriter: log, logLevel: .error,
                          namePath: name, nestingLevel: 0)
                          
        if gpuDeviceCount > 0 {
            queueStack = [devices[1].queues[0]]
        } else {
            queueStack = [Self.syncQueue]
            writeLog("There are no '\(self.name)' devices installed",
                     level: .warning)
        }

        diagnostic(.device, "default: \(queueStack[0].name)",
                    categories: .device)
    }
}

//==============================================================================
/// cpuFallback
/// if `status` is equal to `cudaErrorNotSupported` then a diagnostic message
/// is logged and the fallback body is executed.
@usableFromInline func _cpuFallback(
    _ body: (Platform.Device.Queue) -> Void
) {
    let name = currentQueue.deviceName
    using(device: 0) {
        currentQueue.diagnostic(.fallback,
            "unsupported function on \(name) " +
            "delegated to \(currentQueue.name)",
                categories: .fallback) 
        body(currentQueue)
    }
}

@inlinable public func cpuFallback(
    _ status: cudaError_t,
    _ body: (Platform.Device.Queue) -> Void
) {
    if status == cudaErrorNotSupported {
        _cpuFallback(body)
    } else {
        cudaCheck(status)
    }
}

@inlinable public func cpuFallback(
    _ status: cudnnStatus_t,
    _ body: (Platform.Device.Queue) -> Void
) {
    if status == CUDNN_STATUS_NOT_SUPPORTED {
        _cpuFallback(body)
    } else {
        cudaCheck(status)
    }
}

//==============================================================================
// cudaCheck cudaStream_t
@inlinable public func cudaCheck(
    _ stream: cudaStream_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
#if DEBUG
    cudaStreamSynchronize(stream)
    let status = cudaGetLastError()
    if status != cudaSuccess {
        let location = "CUDA error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudaGetErrorString(status))!
        cudaDeviceReset()
        currentQueue.writeLog("\(message) at \(location)")
        fatalError("unrecoverable error")
    }
#endif
}

//==============================================================================
// cudaCheck cudaError_t
@inlinable public func cudaCheck(
    _ status: cudaError_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    if status != cudaSuccess {
        let location = "CUDA error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudaGetErrorString(status))!
        cudaDeviceReset()
        currentQueue.writeLog("\(message) at \(location)")
        fatalError("unrecoverable error")
    }
}

//==============================================================================
// cudaCheck cudnnStatus_t
@inlinable public func cudaCheck(
    _ status: cudnnStatus_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    if status != CUDNN_STATUS_SUCCESS {
        let location = "CUDNN error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudnnGetErrorString(status))!
        print(message)
        cudaDeviceReset()
        currentQueue.writeLog("\(message) at \(location)")
        fatalError("unrecoverable error")
    }
}

//==============================================================================
// cudaCheck cublasStatus_t
@inlinable public func cudaCheck(
    _ status: cublasStatus_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) {
    if status != CUBLAS_STATUS_SUCCESS {
        let location = "CUBLAS error in \(file) at \(function):\(line)"
        let message = String(utf8String: cublasGetErrorString(status))!
        cudaDeviceReset()
        currentQueue.writeLog("\(message) at \(location)")
        fatalError("unrecoverable error")
    }
}

extension cublasStatus_t : Hashable {}

@inlinable public func cublasGetErrorString(_ status: cublasStatus_t) -> String
{
    let messages = [
        CUBLAS_STATUS_SUCCESS: "CUBLAS_STATUS_SUCCESS",
        CUBLAS_STATUS_NOT_INITIALIZED: "CUBLAS_STATUS_NOT_INITIALIZED",
        CUBLAS_STATUS_ALLOC_FAILED: "CUBLAS_STATUS_ALLOC_FAILED",
        CUBLAS_STATUS_INVALID_VALUE: "CUBLAS_STATUS_INVALID_VALUE",
        CUBLAS_STATUS_ARCH_MISMATCH: "CUBLAS_STATUS_ARCH_MISMATCH",
        CUBLAS_STATUS_MAPPING_ERROR: "CUBLAS_STATUS_MAPPING_ERROR",
        CUBLAS_STATUS_EXECUTION_FAILED: "CUBLAS_STATUS_EXECUTION_FAILED",
        CUBLAS_STATUS_INTERNAL_ERROR: "CUBLAS_STATUS_INTERNAL_ERROR",
        CUBLAS_STATUS_NOT_SUPPORTED: "CUBLAS_STATUS_NOT_SUPPORTED",
        CUBLAS_STATUS_LICENSE_ERROR: "CUBLAS_STATUS_LICENSE_ERROR",
    ]
    return messages[status] ?? "Unknown cublasStatus_t value: \(status)"
}

//==============================================================================
// cudaCheck curandStatus_t
@inlinable public func cudaCheck(
    _ status: curandStatus_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line)
throws {
    if status != CURAND_STATUS_SUCCESS {
        let location = "CURAND error in \(file) at \(function):\(line)"
        let message = String(utf8String: curandGetErrorString(status))!
            + "code=(\(status))"
        cudaDeviceReset()
        throw PlatformError.functionFailure(location: location, message: message)
    }
}

extension curandStatus_t : Hashable {}

@inlinable public func curandGetErrorString(_ status: curandStatus_t) -> String
{
    let messages = [
        CURAND_STATUS_SUCCESS: "CURAND_STATUS_SUCCESS",
        CURAND_STATUS_VERSION_MISMATCH: "CURAND_STATUS_VERSION_MISMATCH",
        CURAND_STATUS_NOT_INITIALIZED: "CURAND_STATUS_NOT_INITIALIZED",
        CURAND_STATUS_ALLOCATION_FAILED: "CURAND_STATUS_ALLOCATION_FAILED",
        CURAND_STATUS_TYPE_ERROR: "CURAND_STATUS_TYPE_ERROR",
        CURAND_STATUS_OUT_OF_RANGE: "CURAND_STATUS_OUT_OF_RANGE",
        CURAND_STATUS_LENGTH_NOT_MULTIPLE: "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
        CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
        CURAND_STATUS_LAUNCH_FAILURE: "CURAND_STATUS_LAUNCH_FAILURE",
        CURAND_STATUS_PREEXISTING_FAILURE: "CURAND_STATUS_PREEXISTING_FAILURE",
        CURAND_STATUS_INITIALIZATION_FAILED: "CURAND_STATUS_INITIALIZATION_FAILED",
        CURAND_STATUS_ARCH_MISMATCH: "CURAND_STATUS_ARCH_MISMATCH",
        CURAND_STATUS_INTERNAL_ERROR: "CURAND_STATUS_INTERNAL_ERROR",
    ]
    return messages[status] ?? "Unknown curandStatus_t value: \(status)"
}

//==============================================================================
extension cudaDataType {
    @inlinable public init(_ type: srtDataType) {
        self = unsafeBitCast(type, to: Self.self)
    }
}

//==============================================================================
// leading dimension for matmul
public extension Tensor {
    @inlinable var leadingDimension: Int {
        assert(Shape.rank == 2 || Shape.rank == 3, "must be rank 2 or 3")
        let i = Shape.rank == 2 ? 0 : 1
        let n = shape[i + 1]
        switch order {
        case .col, .row: return strides[i]
        case .colTiled32: return 32 * shape[i]
        case .colTiledTC32x8: return 32 * n.roundUp(toMultipleOf: 8)
        case .colTiledTC32x32: return 32 * n.roundUp(toMultipleOf: 32)
        default: fatalError("not implemented yet")
        }
    }
}

//==============================================================================
// ReductionType extension
extension cudnnReduceTensorOp_t : Hashable {}

extension ReductionType {
    @inlinable public var cudnn: cudnnReduceTensorOp_t {
        let ops = [
            ReductionType.add: CUDNN_REDUCE_TENSOR_ADD,
            ReductionType.mul: CUDNN_REDUCE_TENSOR_MUL,
            ReductionType.min: CUDNN_REDUCE_TENSOR_MIN,
            ReductionType.max: CUDNN_REDUCE_TENSOR_MAX,
            ReductionType.amax: CUDNN_REDUCE_TENSOR_AMAX,
            ReductionType.mean: CUDNN_REDUCE_TENSOR_AVG,
            ReductionType.asum: CUDNN_REDUCE_TENSOR_NORM1,
            ReductionType.sqrtSumSquares: CUDNN_REDUCE_TENSOR_NORM2,
        ]
        return ops[self]!
    }
}

//------------------------------------------------------------------------------
// NanPropagation
extension NanPropagation {
    @inlinable public var cudnn: cudnnNanPropagation_t {
        switch self {
        case .noPropagate: return CUDNN_NOT_PROPAGATE_NAN
        case .propagate: return CUDNN_PROPAGATE_NAN
        }
    }
}

//------------------------------------------------------------------------------
// TransposeOp
extension TransposeOp {
    @inlinable public var cublas: cublasOperation_t {
        switch self {
        case .noTranspose: return CUBLAS_OP_N
        case .transpose: return CUBLAS_OP_T
        case .hermitian: return CUBLAS_OP_HERMITAN
        case .conjugateTranspose: return CUBLAS_OP_CONJG
        }
    }

    @inlinable public init(_ op: cublasOperation_t) {
        switch op {
        case CUBLAS_OP_N: self = .noTranspose
        case CUBLAS_OP_T: self = .transpose
        case CUBLAS_OP_C, CUBLAS_OP_HERMITAN: self = .hermitian
        case CUBLAS_OP_CONJG: self = .conjugateTranspose
        default: fatalError("unsupported cublasOperation_t")
        }
    }
}

//==============================================================================
/// CublasHandle
/// creates and manages the lifetime of a cublas light handle
public final class CublasHandle 
{
    public let handle: cublasLtHandle_t

    @inlinable public init() {
        var temp: cublasLtHandle_t?
        cudaCheck(cublasLtCreate(&temp))
        handle = temp!
    }

    @inlinable deinit {
        cudaCheck(cublasLtDestroy(handle))
    }
}

//==============================================================================
/// CudnnHandle
/// creates and manages the lifetime of a cudnn handle
public final class CudnnHandle {
    // properties
    public let gpuId: Int
    public let handle: cudnnHandle_t

    //--------------------------------------------------------------------------
    /// init
    /// - Parameters:
    ///  - gpuId: the associated device
    ///  - stream: the associated stream
    @inlinable init(gpuId: Int, using stream: cudaStream_t) {
        self.gpuId = gpuId
        cudaCheck(cudaSetDevice(Int32(gpuId)))

        var temp: cudnnHandle_t?
        cudaCheck(cudnnCreate(&temp))
        handle = temp!
        cudaCheck(cudnnSetStream(handle, stream))
    }

    // deinit
    @inlinable deinit {
        cudaCheck(cudaSetDevice(Int32(gpuId)))
        cudaCheck(cudnnDestroy(handle))
    }
}

//==============================================================================
// DropoutDescriptor
public class DropoutDescriptor {
    // properties
    public let desc: cudnnDropoutDescriptor_t
    public let states: DeviceMemory

    // initializers
    @inlinable public init(
        stream: CudaQueue,
        drop: Double,
        seed: UInt64,
        tensorDesc: TensorDescriptor
    ) {
            // create the descriptor
            var temp: cudnnDropoutDescriptor_t?
            cudaCheck(cudnnCreateDropoutDescriptor(&temp))
            desc = temp!

            // get states size
            var stateSizeInBytes = 0
            cudaCheck(cudnnDropoutGetStatesSize(
                tensorDesc.desc, &stateSizeInBytes))

            // create states array
            states = stream.allocate(stateSizeInBytes)

            // initialize
            cudaCheck(cudnnSetDropoutDescriptor(
                desc,
                stream.cudnn.handle,
                Float(drop),
                states.buffer.baseAddress!,
                states.buffer.count,
                seed
            ))
    }

    @inlinable deinit {
        cudaCheck(cudnnDestroyDropoutDescriptor(desc))
    }
}

//==============================================================================
// FilterDescriptor
public final class FilterDescriptor {
    // properties
    public let desc: cudnnFilterDescriptor_t

    // initializers
    @inlinable public init<S,E>(_ tensor: Tensor<S,E>) {
        // create the descriptor
        var temp: cudnnFilterDescriptor_t?
        cudaCheck(cudnnCreateFilterDescriptor(&temp))
        desc = temp!

        // initialize
        cudaCheck(cudnnSetFilterNdDescriptor(
            desc,
            E.cudnn,
            CUDNN_TENSOR_NHWC,
            Int32(tensor.count),
            tensor.shape.asInt32))
    }

    @inlinable deinit {
        cudaCheck(cudnnDestroyFilterDescriptor(desc))
    }
}

//==============================================================================
/// LRNDescriptor
/// creates and manages the lifetime of a cudnn cudnnLRNDescriptor_t
public final class LRNDescriptor {
    // properties
    public let desc: cudnnLRNDescriptor_t

    // initializers
    @inlinable public init(N: Int, alpha: Double, beta: Double, K: Double) {
        assert(N >= Int(CUDNN_LRN_MIN_N) && N <= Int(CUDNN_LRN_MAX_N),
               "N = \(N) is invalid. Range \(CUDNN_LRN_MIN_N) " +
               "to \(CUDNN_LRN_MAX_N)")
        assert(K >= CUDNN_LRN_MIN_K,
               "K = \(K) is invalid. Must be >= to \(CUDNN_LRN_MIN_K)")
        assert(beta >= CUDNN_LRN_MIN_BETA,
               "beta = \(beta) is invalid. Must be >= to \(CUDNN_LRN_MIN_BETA)")

        // create the descriptor
        var temp: cudnnLRNDescriptor_t?
        cudaCheck(cudnnCreateLRNDescriptor(&temp))
        desc = temp!

        // initialize
        cudaCheck(cudnnSetLRNDescriptor(
            desc, CUnsignedInt(N), alpha, beta, K))
    }

    @inlinable deinit {
        cudaCheck(cudnnDestroyLRNDescriptor(desc))
    }
}

//==============================================================================
/// TensorDescriptor
/// creates and manages the lifetime of a cudnn tensor handle
public final class TensorDescriptor {
    // properties
    public let desc: cudnnTensorDescriptor_t

    //--------------------------------------------------------------------------
    @inlinable public init<S: TensorShape>(
        shape: S,
        strides: S,
        scalarType: cudnnDataType_t
    ) {
        assert(shape.count >= 4 && shape.count <= CUDNN_DIM_MAX,
            "cudnn tensor rank must be between 4 and \(CUDNN_DIM_MAX)")
        // create the descriptor
        var temp: cudnnTensorDescriptor_t?
        cudaCheck(cudnnCreateTensorDescriptor(&temp))
        self.desc = temp!

        // initialize
        cudaCheck(cudnnSetTensorNdDescriptor(
            self.desc,
            scalarType,
            Int32(shape.count),
            shape.asInt32,
            strides.asInt32))
    }

    @inlinable public init(owning desc: cudnnTensorDescriptor_t) {
        self.desc = desc
    }

    //--------------------------------------------------------------------------
    @inlinable deinit {
        cudaCheck(cudnnDestroyTensorDescriptor(desc))
    }

    //--------------------------------------------------------------------------
    // getInfo
    @inlinable public func getInfo()
    -> (extent: [Int], strides: [Int], type: cudnnDataType_t)
    {
        let reqDims = Int(CUDNN_DIM_MAX)
        var dims = [Int32](repeating: 0, count: reqDims)
        var strides = [Int32](repeating: 0, count: reqDims)
        var type = cudnnDataType_t(0)
        var numDims: Int32 = 0

        cudaCheck(cudnnGetTensorNdDescriptor(
            desc,
            Int32(reqDims),
            &type,
            &numDims,
            &dims,
            &strides
        ))

        return (dims[0..<Int(numDims)].map(Int.init),
                strides[0..<Int(numDims)].map(Int.init),
                type)
    }
}

//==============================================================================
/// createTensorDescriptor
/// creates a cudnn tensor descriptor for the associated Tensor
extension Tensor {
    @inlinable public func createTensorDescriptor() -> TensorDescriptor {
        return TensorDescriptor(shape: shape,
                                strides: strides,
                                scalarType: TensorElement.cudnn)
    }
}

//==============================================================================
/// ReductionTensorDescriptor
/// creates and manages the lifetime of a cudnn cudnnLRNDescriptor_t
public final class ReductionTensorDescriptor {
    // properties
    public let desc: cudnnReduceTensorDescriptor_t

    //--------------------------------------------------------------------------
    @inlinable public init(
        type: ReductionType,
        nan: NanPropagation,
        dataType: srtDataType
    ) {
        // create the descriptor
        var temp: cudnnReduceTensorDescriptor_t?
        cudaCheck(cudnnCreateReduceTensorDescriptor(&temp))
        desc = temp!

        let indicesAction = (type == .min || type == .max) ?
            CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
                CUDNN_REDUCE_TENSOR_NO_INDICES

        // initialize
        cudaCheck(cudnnSetReduceTensorDescriptor(
            desc,
            type.cudnn,
            dataType == real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
            nan.cudnn,
            indicesAction,
            CUDNN_32BIT_INDICES
        ))
    }

    //--------------------------------------------------------------------------
    @inlinable deinit {
        cudaCheck(cudnnDestroyReduceTensorDescriptor(desc))
    }
}
