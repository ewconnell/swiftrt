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

//==============================================================================
/// CudaPlatform
/// The collection of compute resources available to the application
/// on the machine where the process is being run.
public class CudaPlatform: Platform {
    // properties
    public static var defaultCpuQueueCount: Int = 1
    public static var defaultAcceleratorQueueCount: Int = 2
    public var discreteMemoryDeviceId: Int = 1
    public var devices: [CudaDevice]
    public let logInfo: LogInfo
    public let name: String
    public var queueStack: [CudaQueue]
    public let syncQueue: CudaQueue

    //--------------------------------------------------------------------------
    // initializer
    @inlinable public init() {
        name = "\(Self.self)"
        logInfo = LogInfo(logWriter: Context.log, logLevel: .error,
                          namePath: name, nestingLevel: 0)

        // make the first queue the sync queue so diagnostics are
        // easier to read. Do this before creating devices.
        let syncQueueId = Context.nextQueueId

        //----------------------------
        // CudaDevice is overloaded to avoid using Swift existentials
        // to support both cpu and gpu operations.
        // Device 0 is the cpu
        let cpuDevice = CudaDevice(deviceId: 0, gpuId: 0, parent: logInfo)
        devices = [cpuDevice]

        syncQueue = CudaQueue(queueId: syncQueueId,
                              parent: cpuDevice.logInfo,
                              gpuDeviceId: cpuDevice.id,
                              deviceName: cpuDevice.name,
                              cpuQueueMode: .sync,
                              useGpu: false)

        //----------------------------
        // query cuda to get number of installed devices
        queueStack = []
        var gpuDeviceCount: CInt = 0
        do {
            try cudaCheck(status: cudaGetDeviceCount(&gpuDeviceCount))
        } catch {
            writeLog("cudaGetDeviceCount failed. " +
                "The Cuda driver may be in an unstable state")
            fatalError()
        }

        // add device for each reported gpu
        for gpuId in 0..<Int(gpuDeviceCount) {
            devices.append(CudaDevice(deviceId: gpuId - 1,
                                      gpuId: gpuId,
                                      parent: logInfo))
        }

        //----------------------------
        // select first gpu queue 0 as default
        if gpuDeviceCount == 0 {
            writeLog("There are no '\(self.name)' devices installed",
                     level: .warning)
            queueStack = [syncQueue]
        } else if devices[0].queues.count > 0 {
            queueStack = [devices[1].queues[0]]
        } else {
            queueStack = [syncQueue]
        }

        //----------------------------
        // report device stats
        if willLog(level: .diagnostic) {
            for device in devices[1...] {
                diagnostic("\(deviceString) \(device.name)", categories: .device)
                diagnostic(" device type       : \(device.properties[.deviceName]!)", categories: .device)
                diagnostic(" global memory     : \(device.properties[.globalMemory]!)", categories: .device)
                diagnostic(" compute capability: \(device.properties[.computeCapability]!)", categories: .device)
                diagnostic(" multiprocessors   : \(device.properties[.multiprocessors]!)", categories: .device)
                diagnostic(" unified addressing: \(device.properties[.unifiedAddressing]!)", categories: .device)
            }
        }
    }
}

//==============================================================================
// cudaCheck cudaError_t
@inlinable public func cudaCheck(
    status: cudaError_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) throws {
    if status != cudaSuccess {
        let location = "CUDA error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudaGetErrorString(status))!
        cudaDeviceReset()
        throw PlatformError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cudnnStatus_t
@inlinable public func cudaCheck(
    status: cudnnStatus_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line)
throws {
    if status != CUDNN_STATUS_SUCCESS {
        let location = "CUDNN error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudnnGetErrorString(status))!
        print(message)
        cudaDeviceReset()
        throw PlatformError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cublasStatus_t
@inlinable public func cudaCheck(
    status: cublasStatus_t,
    file: String = #file,
    function: String = #function,
    line: Int = #line
) throws {
    if status != CUBLAS_STATUS_SUCCESS {
        let location = "CUBLAS error in \(file) at \(function):\(line)"
        let message = String(utf8String: cublasGetErrorString(status))!
            + "code=(\(status))"
        cudaDeviceReset()
        throw PlatformError.functionFailure(location: location, message: message)
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
    status: curandStatus_t,
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
/// ReductionContext
public protocol ReductionContext {
}

//------------------------------------------------------------------------------
// ReductionOp extension
extension cudnnReduceTensorOp_t : Hashable {}

extension ReductionOp {
    @inlinable public var cudnn: cudnnReduceTensorOp_t {
        let ops = [
            ReductionOp.add: CUDNN_REDUCE_TENSOR_ADD,
            ReductionOp.mul: CUDNN_REDUCE_TENSOR_MUL,
            ReductionOp.min: CUDNN_REDUCE_TENSOR_MIN,
            ReductionOp.max: CUDNN_REDUCE_TENSOR_MAX,
            ReductionOp.amax: CUDNN_REDUCE_TENSOR_AMAX,
            ReductionOp.mean: CUDNN_REDUCE_TENSOR_AVG,
            ReductionOp.asum: CUDNN_REDUCE_TENSOR_NORM1,
            ReductionOp.sqrtSumSquares: CUDNN_REDUCE_TENSOR_NORM2,
        ]
        return ops[self]!
    }
}

//------------------------------------------------------------------------------
// ScalarType extension
extension cudnnDataType_t : Hashable {}

extension ScalarType {
    @inlinable public init(cudnn: cudnnDataType_t) {
        let types: [cudnnDataType_t : ScalarType] = [
            CUDNN_DATA_INT8: .real8U,
            CUDNN_DATA_INT32: .real32I,
            CUDNN_DATA_HALF: .real16F,
            CUDNN_DATA_FLOAT: .real32F,
            CUDNN_DATA_DOUBLE: .real64F,
        ]
        assert(types[cudnn] != nil, "Unknown cudnnDataType_t")
        self = types[cudnn]!
    }

    @inlinable public var cudnn: cudnnDataType_t {
        switch self {
        case .real8U: return CUDNN_DATA_INT8
        case .real32I: return CUDNN_DATA_INT32
        case .real16F: return CUDNN_DATA_HALF
        case .real32F: return CUDNN_DATA_FLOAT
        case .real64F: return CUDNN_DATA_DOUBLE
        default: fatalError("Invalid state")
        }
    }

    @inlinable public var cuda: cudaDataType {
        let types: [ScalarType : cudaDataType] = [
            .real16F: CUDA_R_16F,
            .real32F: CUDA_R_32F,
            .real64F: CUDA_R_64F,
            .real8U:  CUDA_R_8U,
            .real32I: CUDA_R_32I,
        ]
        assert(types[self] != nil, "Unknown cudnnDataType_t")
        return types[self]!
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
        case .conjugateTranspose: return CUBLAS_OP_C
        }
    }
}

//==============================================================================
/// CudnnHandle
/// creates and manages the lifetime of a cudnn handle
public final class CudnnHandle {
    // properties
    public let deviceId: Int
    public let handle: cudnnHandle_t

    //--------------------------------------------------------------------------
    /// init
    /// - Parameters:
    ///  - deviceId: the associated device
    ///  - stream: the associated stream
    @inlinable init(deviceId: Int, using stream: cudaStream_t) {
        do {
            self.deviceId = deviceId
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

            var temp: cudnnHandle_t?
            try cudaCheck(status: cudnnCreate(&temp))
            handle = temp!
            try cudaCheck(status: cudnnSetStream(handle, stream))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    // deinit
    @inlinable deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cudnnDestroy(handle))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
    }
}

//==============================================================================
/// CublasHandle
/// creates and manages the lifetime of a cublas handle
public final class CublasHandle {
    // properties
    public let deviceId: Int
    public let handle: cublasHandle_t

    //--------------------------------------------------------------------------
    /// init
    /// - Parameters:
    ///  - deviceId: the associated device
    ///  - stream: the associated stream
    @inlinable public init(deviceId: Int, using stream: cudaStream_t) {
        do {
            self.deviceId = deviceId
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

            var temp: cublasHandle_t?
            try cudaCheck(status: cublasCreate_v2(&temp))
            handle = temp!
            try cudaCheck(status: cublasSetStream_v2(handle, stream))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    // deinit
    @inlinable deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cublasDestroy_v2(handle))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
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
        do {
            // create the descriptor
            var temp: cudnnDropoutDescriptor_t?
            try cudaCheck(status: cudnnCreateDropoutDescriptor(&temp))
            desc = temp!

            // get states size
            var stateSizeInBytes = 0
            try cudaCheck(status: cudnnDropoutGetStatesSize(
                tensorDesc.desc, &stateSizeInBytes))

            // create states array
            states = try stream.allocate(byteCount: stateSizeInBytes)

            // initialize
            try cudaCheck(status: cudnnSetDropoutDescriptor(
                desc,
                stream.cudnn.handle,
                Float(drop),
                states.buffer.baseAddress!,
                states.buffer.count,
                seed
            ))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyDropoutDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
    }
}

//==============================================================================
// FilterDescriptor
public final class FilterDescriptor {
    // properties
    public let desc: cudnnFilterDescriptor_t

    // initializers
    @inlinable public init<S,E: ScalarElement>(_ tensor: Tensor<S,E>) {
        do {
            // create the descriptor
            var temp: cudnnFilterDescriptor_t?
            try cudaCheck(status: cudnnCreateFilterDescriptor(&temp))
            desc = temp!

            // initialize
            try cudaCheck(status: cudnnSetFilterNdDescriptor(
                desc,
                E.type.cudnn,
                CUDNN_TENSOR_NHWC,
                Int32(tensor.count),
                tensor.shape.asDeviceIndex))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyFilterDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
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
        do {
            guard N >= Int(CUDNN_LRN_MIN_N) && N <= Int(CUDNN_LRN_MAX_N) else {
                throw PlatformError.rangeError(
                    "N = \(N) is invalid. Range \(CUDNN_LRN_MIN_N) " +
                            "to \(CUDNN_LRN_MAX_N)")
            }
            guard K >= CUDNN_LRN_MIN_K else {
                throw PlatformError.rangeError(
                    "K = \(K) is invalid. Must be >= to \(CUDNN_LRN_MIN_K)")
            }
            guard beta >= CUDNN_LRN_MIN_BETA else {
                throw PlatformError.rangeError(
                    "beta = \(beta) is invalid. Must be >= to \(CUDNN_LRN_MIN_BETA)")
            }

            // create the descriptor
            var temp: cudnnLRNDescriptor_t?
            try cudaCheck(status: cudnnCreateLRNDescriptor(&temp))
            desc = temp!

            // initialize
            try cudaCheck(status: cudnnSetLRNDescriptor(
                desc, CUnsignedInt(N), alpha, beta, K))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
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
        scalarType: ScalarType
    ) {
        do {
            // create the descriptor
            var temp: cudnnTensorDescriptor_t?
            try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
            self.desc = temp!

            // initialize
            try cudaCheck(status: cudnnSetTensorNdDescriptor(
                self.desc,
                scalarType.cudnn,
                Int32(shape.count),
                shape.asDeviceIndex,
                strides.asDeviceIndex))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }

    @inlinable public init(owning desc: cudnnTensorDescriptor_t) {
        self.desc = desc
    }

    //--------------------------------------------------------------------------
    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyTensorDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
    }

    //--------------------------------------------------------------------------
    // getInfo
    @inlinable public func getInfo()
    -> (extent: [Int], strides: [Int], ScalarType)
    {
        let reqDims = Int(CUDNN_DIM_MAX)
        var dims = [Int32](repeating: 0, count: reqDims)
        var strides = [Int32](repeating: 0, count: reqDims)
        var type = cudnnDataType_t(0)
        var numDims: Int32 = 0

        do {
            try cudaCheck(status: cudnnGetTensorNdDescriptor(
                desc,
                Int32(reqDims),
                &type,
                &numDims,
                &dims,
                &strides
            ))

            return (dims[0..<Int(numDims)].map { Int($0) },
                    strides[0..<Int(numDims)].map { Int($0) },
                    ScalarType(cudnn: type))
        } catch {
            Context.currentQueue.writeLog("\(createString) \(error)")
            fatalError()
        }
    }
}

//==============================================================================
/// createTensorDescriptor
/// creates a cudnn tensor descriptor for the associated Tensor
extension Tensor where TensorElement: ScalarElement {
    @inlinable public func createTensorDescriptor(
        asShape newShape: Shape? = nil
    ) -> TensorDescriptor {
        assert(newShape == nil || newShape!.count == shape.count)
        return TensorDescriptor(shape: newShape ?? shape,
                                strides: strides,
                                scalarType: TensorElement.type)
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
        op: ReductionOp,
        nan: NanPropagation,
        scalarType: ScalarType
    ) throws {
        // create the descriptor
        var temp: cudnnReduceTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
        desc = temp!

        let indicesAction = (op == .min || op == .max) ?
            CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
                CUDNN_REDUCE_TENSOR_NO_INDICES

        // initialize
        try cudaCheck(status: cudnnSetReduceTensorDescriptor(
            desc,
            op.cudnn,
            scalarType == .real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
            nan.cudnn,
            indicesAction,
            CUDNN_32BIT_INDICES
        ))
    }

    //--------------------------------------------------------------------------
    @inlinable deinit {
        do {
            try cudaCheck(status: cudnnDestroyReduceTensorDescriptor(desc))
        } catch {
            Context.currentQueue.writeLog(
                "\(releaseString) \(Self.self) \(error)")
        }
    }
}
