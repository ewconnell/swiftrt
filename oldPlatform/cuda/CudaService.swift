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

//==============================================================================
public let cudaServiceName = "cuda"

extension Platform {
    // shortcut to cuda sercoe
    public static var cuda: CudaService? = {
        return Platform.local.services[cudaServiceName] as? CudaService
    }()
    
    /// cudaConfiguration
    /// used to override default cpu configuration
    static var cudaConfiguration: [CudaPropertyKey: Any] = [
        .queuesPerDevice: 2
    ]
}

//==============================================================================
/// a set of predefined property names to simplify configuring
/// the service properties
public enum CudaPropertyKey: Int {
    case queuesPerDevice
}

//==============================================================================
// CudaService
public final class CudaService: LocalComputeService {
    // properties
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String

    //--------------------------------------------------------------------------
    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach {
                $0.timeout = timeout
            }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "cuda"
        self.logInfo = logInfo
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)
        
        // create devices
        var deviceCount: CInt = 0
        do {
            try cudaCheck(status: cudaGetDeviceCount(&deviceCount))
        } catch {
            writeLog("cudaGetDeviceCount failed. " +
                "The Cuda driver may be in an unstable state",
                     level: .error)
            throw error
        }
        
        guard deviceCount > 0 else {
            writeLog("There are no '\(self.name)' devices installed",
                level: .warning)
            throw ServiceError.serviceIsUnavailable
        }
        
        // add device object for each id reported
        for i in 0..<Int(deviceCount) {
            let device = try CudaDevice(service: self,
                                        deviceId: i,
                                        logInfo: logInfo,
                                        isUnified: false,
                                        timeout: timeout)
            devices.append(device)
        }
    }
    
    deinit {
        ObjectTracker.global.remove(trackingId: trackingId)
    }
}

//==============================================================================
// cudaCheck cudaError_t
public func cudaCheck(status: cudaError_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != cudaSuccess {
        let location = "CUDA error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudaGetErrorString(status))!
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cudnnStatus_t
public func cudaCheck(status: cudnnStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CUDNN_STATUS_SUCCESS {
        let location = "CUDNN error in \(file) at \(function):\(line)"
        let message = String(utf8String: cudnnGetErrorString(status))!
        print(message)
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

//==============================================================================
// cudaCheck cublasStatus_t
public func cudaCheck(status: cublasStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CUBLAS_STATUS_SUCCESS {
        let location = "CUBLAS error in \(file) at \(function):\(line)"
        let message = String(utf8String: cublasGetErrorString(status))!
            + "code=(\(status))"
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

extension cublasStatus_t : Hashable {}

public func cublasGetErrorString(_ status: cublasStatus_t) -> String {
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
public func cudaCheck(status: curandStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
    if status != CURAND_STATUS_SUCCESS {
        let location = "CURAND error in \(file) at \(function):\(line)"
        let message = String(utf8String: curandGetErrorString(status))!
            + "code=(\(status))"
        cudaDeviceReset()
        throw ServiceError.functionFailure(location: location, message: message)
    }
}

extension curandStatus_t : Hashable {}

public func curandGetErrorString(_ status: curandStatus_t) -> String {
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
    public var cudnn: cudnnReduceTensorOp_t {
        get {
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
}

//==============================================================================
// TensorFormat
extension cudnnTensorFormat_t : Hashable {}

extension TensorFormat {
    public var cudnn: cudnnTensorFormat_t {
        get {
            let formats: [TensorFormat : cudnnTensorFormat_t] = [
                .scalar: CUDNN_TENSOR_NHWC,
                .vector: CUDNN_TENSOR_NHWC,
                .matrix: CUDNN_TENSOR_NHWC,
                .volume: CUDNN_TENSOR_NHWC,
                .nchw: CUDNN_TENSOR_NCHW,
                .nhwc: CUDNN_TENSOR_NHWC,
            ]
            assert(formats[self] != nil,
                   "TensorFormat: \(self) not supported by cudnn")
            return formats[self]!
        }
    }
}

//------------------------------------------------------------------------------
// ScalarType extension
extension cudnnDataType_t : Hashable {}

extension ScalarType {
    public init(cudnn: cudnnDataType_t) {
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

    public var cudnn: cudnnDataType_t {
        get {
            switch self {
            case .real8U: return CUDNN_DATA_INT8
            case .real32I: return CUDNN_DATA_INT32
            case .real16F: return CUDNN_DATA_HALF
            case .real32F: return CUDNN_DATA_FLOAT
            case .real64F: return CUDNN_DATA_DOUBLE
            default: fatalError("Invalid state")
            }
        }
    }

    public var cuda: cudaDataType {
        get {
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
}

//------------------------------------------------------------------------------
// NanPropagation
extension NanPropagation {
    public var cudnn: cudnnNanPropagation_t {
        get {
            switch self {
            case .noPropagate: return CUDNN_NOT_PROPAGATE_NAN
            case .propagate: return CUDNN_PROPAGATE_NAN
            }
        }
    }
}

//------------------------------------------------------------------------------
// TransposeOp
extension TransposeOp {
    public var cublas: cublasOperation_t {
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
public final class CudnnHandle: ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    private let deviceId: Int
    public var handle: cudnnHandle_t

    //--------------------------------------------------------------------------
    /// init
    /// - Parameter deviceId:
    /// - Parameter using:
    /// - Parameter isStatic: true if the handle belongs to a statically
    ///   held stream and the lifetime should not be tracked
    init(deviceId: Int, using stream: cudaStream_t, isStatic: Bool) throws {
        self.deviceId = deviceId
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

        var temp: cudnnHandle_t?
        try cudaCheck(status: cudnnCreate(&temp))
        handle = temp!
        try cudaCheck(status: cudnnSetStream(handle, stream))
        trackingId = ObjectTracker.global.register(self, isStatic: isStatic)
    }

    //--------------------------------------------------------------------------
    // deinit
    deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cudnnDestroy(handle))
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            print("\(releaseString) CudnnHandle(\(trackingId)) "
                + "\(String(describing: error))")
        }
    }
}

//==============================================================================
/// CublasHandle
/// creates and manages the lifetime of a cublas handle
public final class CublasHandle: ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    private let deviceId: Int
    public var handle: cublasHandle_t

    //--------------------------------------------------------------------------
    /// init
    /// - Parameter deviceId:
    /// - Parameter using:
    /// - Parameter isStatic: true if the handle belongs to a statically
    ///   held stream and the lifetime should not be tracked
    public init(deviceId: Int, using stream: cudaStream_t,
                isStatic: Bool) throws
    {
        self.deviceId = deviceId
        try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

        var temp: cublasHandle_t?
        try cudaCheck(status: cublasCreate_v2(&temp))
        handle = temp!
        try cudaCheck(status: cublasSetStream_v2(handle, stream))
        trackingId = ObjectTracker.global.register(self, isStatic: isStatic)
    }

    //--------------------------------------------------------------------------
    // deinit
    deinit {
        do {
            try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
            try cudaCheck(status: cublasDestroy_v2(handle))
            ObjectTracker.global.remove(trackingId: trackingId)
        } catch {
            print("\(releaseString) CublasHandle(\(trackingId)) "
                + "\(String(describing: error))")
        }
    }
}

//==============================================================================
// ActivationDescriptor
public final class ActivationDescriptor : ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnActivationDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
    public init(mode: ActivationMode,
                nan: NanPropagation,
                reluCeiling: Double) throws
    {
		// create the descriptor
		var temp: cudnnActivationDescriptor_t?
		try cudaCheck(status: cudnnCreateActivationDescriptor(&temp))
		desc = temp!

		// initialize
		try cudaCheck(status: cudnnSetActivationDescriptor(
			desc, mode.cudnn, nan.cudnn, reluCeiling))
		trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // deinit
	deinit {
		try! cudaCheck(status: cudnnDestroyActivationDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}
}

//==============================================================================
// ConvolutionDescriptor
public final class ConvolutionDescriptor : ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnConvolutionDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
	public init(scalarType: ScalarType,
                rank: Int,
                padding: [Int],
	            strides: [Int],
                dilations: [Int],
                mode: ConvolutionMode) throws
    {
		// create the descriptor
		var temp: cudnnConvolutionDescriptor_t?
		try cudaCheck(status: cudnnCreateConvolutionDescriptor(&temp))
		desc = temp!

		// initialize
		try cudaCheck(status: cudnnSetConvolutionNdDescriptor(
			desc,
            Int32(rank),
			padding.map { Int32($0) },
			strides.map { Int32($0) },
			dilations.map { Int32($0) },
			mode.cudnn,
			scalarType.cudnn))

		trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // deinit
	deinit {
		try! cudaCheck(status: cudnnDestroyConvolutionDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}
}

//==============================================================================
// DropoutDescriptor
public final class DropoutDescriptor: ObjectTracking {
    // properties
    private var states: DeviceArray
    public private (set) var trackingId = 0
    let desc: cudnnDropoutDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
	public init(stream: CudaQueue, drop: Double, seed: UInt64,
	            tensorDesc: TensorDescriptor) throws {
		// create the descriptor
		var temp: cudnnDropoutDescriptor_t?
		try cudaCheck(status: cudnnCreateDropoutDescriptor(&temp))
		desc = temp!

		// get states size
		var stateSizeInBytes = 0
		try cudaCheck(status: cudnnDropoutGetStatesSize(
			tensorDesc.desc, &stateSizeInBytes))

		// create states array
		states = try stream.device.createArray(byteCount: stateSizeInBytes,
                                               heapIndex: 0, zero: false)

		// initialize
		try cudaCheck(status: cudnnSetDropoutDescriptor(
			desc,
			stream.cudnn.handle,
			Float(drop),
            states.buffer.baseAddress!,
            states.buffer.count,
			seed
		))

		trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // cleanup
	deinit {
		try! cudaCheck(status: cudnnDestroyDropoutDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}
}

//==============================================================================
// FilterDescriptor
public final class FilterDescriptor : ObjectTracking {
	// initializers
    public init<T>(_ tensor: T) throws where
        T: TensorView, T.Element: AnyNumeric
    {
		// create the descriptor
		var temp: cudnnFilterDescriptor_t?
		try cudaCheck(status: cudnnCreateFilterDescriptor(&temp))
		desc = temp!

		// initialize
		try cudaCheck(status: cudnnSetFilterNdDescriptor(
            desc,
            T.Element.scalarType.cudnn,
			tensor.format.cudnn,
			Int32(tensor.extents.count),
			tensor.extents.map { Int32($0)}))

		trackingId = ObjectTracker.global.register(self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyFilterDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}

	// properties
	public private (set) var trackingId = 0
	let desc: cudnnFilterDescriptor_t
}

//==============================================================================
/// LRNDescriptor
/// creates and manages the lifetime of a cudnn cudnnLRNDescriptor_t
public final class LRNDescriptor: ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnLRNDescriptor_t

    //--------------------------------------------------------------------------
	// initializers
	public init(N: Int, alpha: Double, beta: Double, K: Double) throws {
		guard N >= Int(CUDNN_LRN_MIN_N) && N <= Int(CUDNN_LRN_MAX_N) else {
			throw ServiceError.rangeError(
				"N = \(N) is invalid. Range \(CUDNN_LRN_MIN_N) " +
                        "to \(CUDNN_LRN_MAX_N)")
		}
		guard K >= CUDNN_LRN_MIN_K else {
			throw ServiceError.rangeError(
				"K = \(K) is invalid. Must be >= to \(CUDNN_LRN_MIN_K)")
		}
		guard beta >= CUDNN_LRN_MIN_BETA else {
			throw ServiceError.rangeError(
				"beta = \(beta) is invalid. Must be >= to \(CUDNN_LRN_MIN_BETA)")
		}

		// create the descriptor
		var temp: cudnnLRNDescriptor_t?
		try cudaCheck(status: cudnnCreateLRNDescriptor(&temp))
		desc = temp!

		// initialize
		try cudaCheck(status: cudnnSetLRNDescriptor(
            desc, CUnsignedInt(N), alpha, beta, K))

        trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // cleanup
	deinit {
		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}
}

//==============================================================================
/// TensorDescriptor
/// creates and manages the lifetime of a cudnn tensor handle
public final class TensorDescriptor: ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnTensorDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
    public init(shape: DataShape, scalarType: ScalarType) throws {
        // create the descriptor
        var temp: cudnnTensorDescriptor_t?
        try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
        self.desc = temp!

        // initialize
        try cudaCheck(status: cudnnSetTensorNdDescriptor(
            self.desc,
            scalarType.cudnn,
            Int32(shape.extents.count),
            shape.extents.map { Int32($0) },
            shape.strides.map { Int32($0) }))

        trackingId = ObjectTracker.global.register(self)
    }

    public init(owning desc: cudnnTensorDescriptor_t) {
        self.desc = desc
        trackingId = ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    // cleanup
    deinit {
        try! cudaCheck(status: cudnnDestroyTensorDescriptor(desc))
        ObjectTracker.global.remove(trackingId: trackingId)
    }

    //--------------------------------------------------------------------------
    // getInfo
    public func getInfo() throws -> (extent: [Int], strides: [Int], ScalarType) {
        let reqDims = Int(CUDNN_DIM_MAX)
        var dims = [Int32](repeating: 0, count: reqDims)
        var strides = [Int32](repeating: 0, count: reqDims)
        var type = cudnnDataType_t(0)
        var numDims: Int32 = 0

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
    }
}

//==============================================================================
/// createTensorDescriptor
/// creates a cudnn tensor descriptor for the associated TensorView
extension TensorView where Element: AnyNumeric {
	public func createTensorDescriptor(
        asShape newShape: DataShape? = nil) throws -> TensorDescriptor
    {
		assert(newShape == nil || newShape!.elementCount == shape.elementCount)
        return try TensorDescriptor(shape: newShape ?? shape,
                                    scalarType: Element.scalarType)
	}
}

//==============================================================================
// PoolingDescriptor
public final class PoolingDescriptor : ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnPoolingDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
	public init(mode: PoolingMode,
                nan: NanPropagation,
                filterSize: [Int],
	            padding: [Int],
                strides: [Int]) throws
    {
        // validate
        assert(
            filterSize.count == padding.count &&
            filterSize.count == strides.count,
            "filterSize, padding, and strides must have equal counts")

        // create the descriptor
		var temp: cudnnPoolingDescriptor_t?
		try cudaCheck(status: cudnnCreatePoolingDescriptor(&temp))
		desc = temp!

		// initialize
		try cudaCheck(status: cudnnSetPoolingNdDescriptor(
			desc, mode.cudnn, nan.cudnn,
            Int32(filterSize.count),
			filterSize.map { Int32($0) },
			padding.map { Int32($0) },
			strides.map { Int32($0) }))

		trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // cleanup
	deinit {
		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
		ObjectTracker.global.remove(trackingId: trackingId)
	}
}

//==============================================================================
/// ReductionTensorDescriptor
/// creates and manages the lifetime of a cudnn cudnnLRNDescriptor_t
public final class ReductionTensorDescriptor : ObjectTracking {
    // properties
    public private (set) var trackingId = 0
    let desc: cudnnReduceTensorDescriptor_t

    //--------------------------------------------------------------------------
	// initializers
	public init(op: ReductionOp, nan: NanPropagation, scalarType: ScalarType) throws {
		// create the descriptor
		var temp: cudnnReduceTensorDescriptor_t?
		try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
		desc = temp!

		let indicesAction = (op == .min || op == .max) ?
			CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES

		// initialize
		try cudaCheck(status: cudnnSetReduceTensorDescriptor(
			desc,
			op.cudnn,
			scalarType == .real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
			nan.cudnn,
			indicesAction,
			CUDNN_32BIT_INDICES
		))

		trackingId = ObjectTracker.global.register(self)
	}

    //--------------------------------------------------------------------------
    // cleanup
	deinit {
		try! cudaCheck(status: cudnnDestroyReduceTensorDescriptor(desc))
        ObjectTracker.global.remove(trackingId: trackingId)
	}
}
