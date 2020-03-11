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
import CCuda


//==============================================================================
// 
public extension CudaQueue {
    func createConvolutionInferring<T>(
        x: T,
        yShape: inout DataShape,
        filter: T,
        bias: T,
        activation: ActivationMode,
        strides: [Int],
        padding: [Int],
        dilations: [Int],
        properties: ConvolutionProperties) throws -> ConvolutionInferring<T>
        where T: TensorView, T.Element: AnyFloatingPoint
    {
        fatalError("cpu not implemented")
    }
    
    func createConvolutionTraining<T>(
        x: T,
        yShape: inout DataShape,
        filter: T,
        bias: T,
        activation: ActivationMode,
        strides: [Int],
        padding: [Int],
        dilations: [Int],
        properties: ConvolutionProperties) throws -> ConvolutionTraining<T>
        where T: TensorView, T.Element: AnyFloatingPoint
    {
        fatalError("cpu not implemented")
    }
}

//==============================================================================
// CudaConvolution
public struct CudaConvolution<T>: Logging where
    T: TensorView, T.Element: AnyFloatingPoint
{
    // queues
    private let dataQueue: CudaQueue
    private let filterBiasBackQueue: CudaQueue

    // descriptors
    private let convolutionDescriptor: ConvolutionDescriptor
    private var activationDescriptor: ActivationDescriptor
    private let xTensorDescriptor: TensorDescriptor
    private let yTensorDescriptor: TensorDescriptor
    private let filterDescriptor: FilterDescriptor
    private let biasTensorDescriptor: TensorDescriptor

    // forward
    private var fwdAlgo: cudnnConvolutionFwdAlgo_t
    private var fwdWorkspaceSize = 0
    private var fwdWorkspace: DeviceArray?

    // backward data
    private var bwdDataAlgo: cudnnConvolutionBwdDataAlgo_t
    private var bwdDataWorkspaceSize = 0
    private var bwdDataWorkspace: DeviceArray?

    // backward filter
    private var bwdFilterAlgo: cudnnConvolutionBwdFilterAlgo_t
    private var bwdFilterWorkspaceSize = 0
    private var bwdFilterWorkspace: DeviceArray?
    private let logCategories: LogCategories = [.initialize]
    
    //--------------------------------------------------------------------------
    // initializer
    public init(x: T,
                yShape: inout DataShape,
                filter: T,
                bias: T,
                activation: ActivationMode,
                strides: [Int],
                padding: [Int],
                dilations: [Int],
                properties: ConvolutionProperties,
                device: CudaDevice,
                filterBiasBackQueueIndex: Int) throws
    {
        //----------------------------------
        // queues
        // TODO: change this when devices have fixed collections of queues
        // initialization can create workspaces on the devices
        // associated with the queues, so we hold on to them
        let defaultQueue = DeviceContext.currentQueue as! CudaQueue
        self.dataQueue = defaultQueue
        self.filterBiasBackQueue = defaultQueue
        
        xTensorDescriptor = try x.createTensorDescriptor()
        filterDescriptor = try FilterDescriptor(filter)
        biasTensorDescriptor = try bias.createTensorDescriptor()

        //----------------------------------
        // set some initial values so we can use `self` during init
        fwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0

        //----------------------------------
        // create activation descriptor
        activationDescriptor = try ActivationDescriptor(
            mode: activation,
            nan: properties.activationNan,
            reluCeiling: properties.activationReluCeiling)

        //----------------------------------
        // create convolution descriptor
        let convolutionScalarType: ScalarType =
            T.Element.scalarType == .real64F ? .real64F : .real32F

        convolutionDescriptor = try ConvolutionDescriptor(
            scalarType: convolutionScalarType,
            rank: x.rank - 2,
            padding: padding,
            strides: strides,
            dilations: dilations,
            mode: properties.mode)

        //----------------------------------
        // get the extents for the output
        var yExtent = [Int32](repeating: 0, count: x.rank)
        try cudaCheck(status: cudnnGetConvolutionNdForwardOutputDim(
            convolutionDescriptor.desc,
            xTensorDescriptor.desc,
            filterDescriptor.desc,
            Int32(yExtent.count),
            &yExtent))

        //----------------------------------
        // return the shape of the output y and create a tensorDescriptor
        // with the same scalarType for y as x
        yShape = DataShape(extents: yExtent.map { Int($0) })
        yTensorDescriptor = try x.createTensorDescriptor(asShape: yShape)

        try selectForwardAlgorithm(x: x, properties: properties)
       
        // TODO: get this from Context
        // if _Context.evaluationMode == .training {
        try selectBackwardAlgorithm(x: x, properties: properties)
        // }
    }
    
    //--------------------------------------------------------------------------
    // inferring
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
    public func inferring(y: inout T, from x: T, filter: T, bias: T) throws {
        try cudaCheck(status: cudnnConvolutionBiasActivationForward(
            dataQueue.cudnn.handle,
            // alpha1
            T.Element.onePointer,
            // x
            xTensorDescriptor.desc,
            x.deviceReadOnly(using: dataQueue),
            // filter weights
            filterDescriptor.desc,
            filter.deviceReadOnly(using: dataQueue),
            // convDesc
            convolutionDescriptor.desc,
            // algo
            fwdAlgo,
            // workspace device array
            fwdWorkspace?.buffer.baseAddress!,
            // workspace size in bytes
            fwdWorkspaceSize,
            // alpha2
            T.Element.zeroPointer,
            // z used for activation (TODO: inplace on y?? find out what's right)
            yTensorDescriptor.desc,
            y.deviceReadOnly(using: dataQueue),
            // bias
            biasTensorDescriptor.desc,
            bias.deviceReadOnly(using: dataQueue),
            // activation
            activationDescriptor.desc,
            // y
            yTensorDescriptor.desc,
            y.deviceReadWrite(using: dataQueue)))
    }

    //--------------------------------------------------------------------------
    // gradient
    // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
    public func gradient(y: T, yDiff: T,
                         filter: T, filterDiff: inout T,
                         bias: T, biasDiff: inout T,
                         x: T, xDiff: inout T) throws
    {
        // data
        try cudaCheck(status: cudnnConvolutionBackwardData(
            dataQueue.cudnn.handle,
            // alpha
            T.Element.onePointer,
            // filter
            filterDescriptor.desc,
            filter.deviceReadOnly(using: dataQueue),
            // yDiff
            yTensorDescriptor.desc,
            yDiff.deviceReadOnly(using: dataQueue),
            // conv
            convolutionDescriptor.desc,
            // algo
            bwdDataAlgo,
            // workspace
            bwdDataWorkspace?.buffer.baseAddress!,
            bwdDataWorkspaceSize,
            // beta
            T.Element.zeroPointer,
            // xDiff
            xTensorDescriptor.desc,
            xDiff.deviceReadWrite(using: dataQueue)))
        
        // filter
        try cudaCheck(status: cudnnConvolutionBackwardFilter(
            filterBiasBackQueue.cudnn.handle,
            // alpha
            T.Element.onePointer,
            // x
            xTensorDescriptor.desc,
            x.deviceReadOnly(using: dataQueue),
            // yDiff
            yTensorDescriptor.desc,
            yDiff.deviceReadOnly(using: dataQueue),
            // conv
            convolutionDescriptor.desc,
            // algo
            bwdFilterAlgo,
            // workspace
            bwdFilterWorkspace?.buffer.baseAddress!,
            bwdFilterWorkspaceSize,
            // beta
            T.Element.zeroPointer,
            // filterDiff
            filterDescriptor.desc,
            filterDiff.deviceReadWrite(using: dataQueue)))

        // bias
        try cudaCheck(status: cudnnConvolutionBackwardBias(
            filterBiasBackQueue.cudnn.handle,
            // alpha
            T.Element.onePointer,
            // yDiff
            yTensorDescriptor.desc,
            yDiff.deviceReadOnly(using: dataQueue),
            // beta
            T.Element.zeroPointer,
            //
            biasTensorDescriptor.desc,
            biasDiff.deviceReadWrite(using: dataQueue)))
    }

    //--------------------------------------------------------------------------
    // selectForwardAlgorithm
    private mutating func selectForwardAlgorithm(
        x: T, properties: ConvolutionProperties) throws
    {
        switch properties.forwardAlgorithm {
        case .deterministic:
            let algs = try findForwardAlgorithms(x: x)
            var notFound = true
            for alg in algs {
                if alg.determinism == CUDNN_DETERMINISTIC {
                    notFound = false
                    fwdAlgo = alg.algo
                    fwdWorkspaceSize = alg.memory
                    break
                }
            }
            
            // default to the fastest
            if notFound {
                fwdAlgo = algs[0].algo
                fwdWorkspaceSize = algs[0].memory
                writeLog("failed to find 'deterministic' forward " +
                    "convolution algorithm. 'fastest' used instead")
                fallthrough
            }
            
        case .fastest:
            let algs = try findForwardAlgorithms(x: x)
            fwdAlgo = algs[0].algo
            fwdWorkspaceSize = algs[0].memory
            
        case .noWorkspace:
            let algs = try findForwardAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory == 0 { algIndex = i; break }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find 'noWorkspace' forward " +
                    "convolution algorithm")
                throw DeviceError.initializeFailed
            }
            fwdAlgo = algs[algIndex].algo
            fwdWorkspaceSize = algs[algIndex].memory
            
        case .workspaceLimit:
            let algs = try findForwardAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory <= properties.forwardWorkspaceLimit {
                    algIndex = i; break
                }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find suitable 'workspaceLimit' " +
                    "forward convolution algorithm")
                throw DeviceError.initializeFailed
            }
            fwdAlgo = algs[algIndex].algo
            fwdWorkspaceSize = algs[algIndex].memory
            
        default:
            // user explicitly specifies
            fwdAlgo = properties.forwardAlgorithm.cudnn
            
            // get the workspace size
            try cudaCheck(status: cudnnGetConvolutionForwardWorkspaceSize(
                dataQueue.cudnn.handle,
                xTensorDescriptor.desc,
                filterDescriptor.desc,
                convolutionDescriptor.desc,
                yTensorDescriptor.desc,
                fwdAlgo,
                &fwdWorkspaceSize))
        }
        
        // allocate workspace
        if fwdWorkspaceSize > 0 {
            fwdWorkspace = try dataQueue.device
                .createArray(byteCount: fwdWorkspaceSize,
                             heapIndex: 0, zero: false)
        }
        
        // report selection
        let alg = ConvolutionFwdAlgorithm(cudnn: fwdAlgo)
        
        if willLog(level: .diagnostic) && properties.forwardAlgorithm != alg {
            diagnostic("using forward algorithm: " +
                "\(alg)  workspace size: \(fwdWorkspaceSize)",
                categories: logCategories)
        }
    }

    //--------------------------------------------------------------------------
    // selectBackwardAlgorithm
    private mutating func selectBackwardAlgorithm(
        x: T, properties: ConvolutionProperties) throws
    {
        switch properties.backwardDataAlgorithm {
        case .deterministic:
            let algs = try findBackwardDataAlgorithms(x: x)
            var notFound = true
            for alg in algs {
                if alg.determinism == CUDNN_DETERMINISTIC {
                    notFound = false
                    bwdDataAlgo = alg.algo
                    bwdDataWorkspaceSize = alg.memory
                    break
                }
            }

            // default to the fastest
            if notFound {
                bwdDataAlgo = algs[0].algo
                bwdDataWorkspaceSize = algs[0].memory
                writeLog("failed to find 'deterministic' backward data " +
                    "convolution algorithm. 'fastest' used instead")
                fallthrough
            }

        case .fastest:
            let algs = try findBackwardDataAlgorithms(x: x)
            bwdDataAlgo = algs[0].algo
            bwdDataWorkspaceSize = algs[0].memory
            
        case .noWorkspace:
            let algs = try findBackwardDataAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory == 0 { algIndex = i; break }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find 'noWorkspace' backward data " +
                    "convolution algorithm")
                throw DeviceError.initializeFailed
            }
            bwdDataAlgo = algs[algIndex].algo
            bwdDataWorkspaceSize = algs[algIndex].memory
            
        case .workspaceLimit:
            let algs = try findBackwardDataAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory <= properties.backwardDataWorkspaceLimit {
                    algIndex = i; break
                }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find suitable 'workspaceLimit' " +
                    "backward data convolution algorithm")
                throw DeviceError.initializeFailed
            }
            bwdDataAlgo = algs[algIndex].algo
            bwdDataWorkspaceSize = algs[algIndex].memory
            
        default:
            // user explicitly specifies
            bwdDataAlgo = properties.backwardDataAlgorithm.cudnn
            
            // get the workspace size
            try cudaCheck(status: cudnnGetConvolutionBackwardDataWorkspaceSize(
                dataQueue.cudnn.handle,
                filterDescriptor.desc,
                yTensorDescriptor.desc,
                convolutionDescriptor.desc,
                xTensorDescriptor.desc,
                bwdDataAlgo,
                &bwdDataWorkspaceSize))
        }
        
        // allocate workspace
        if bwdDataWorkspaceSize > 0 {
            bwdDataWorkspace =
                try dataQueue.device.createArray(byteCount: bwdDataWorkspaceSize,
                                                 heapIndex: 0, zero: false)
        }

        // report selection
        let dataAlg = ConvolutionBwdDataAlgorithm(cudnn: bwdDataAlgo)

        if willLog(level: .diagnostic) &&
            properties.backwardDataAlgorithm != dataAlg
        {
            diagnostic("using backward data algorithm: " +
                "\(dataAlg)  workspace size: \(bwdDataWorkspaceSize)",
                categories: logCategories)
        }

        //----------------------------------
        // choose best backward filter algorithm
        switch properties.backwardFilterAlgorithm {
        case .deterministic:
            let algs = try findBackwardFilterAlgorithms(x: x)
            var notFound = true
            for alg in algs {
                if alg.determinism == CUDNN_DETERMINISTIC {
                    notFound = false
                    bwdFilterAlgo = alg.algo
                    bwdFilterWorkspaceSize = alg.memory
                    break
                }
            }

            // default to the fastest
            if notFound {
                bwdFilterAlgo = algs[0].algo
                bwdFilterWorkspaceSize = algs[0].memory
                writeLog("failed to find 'deterministic' backward filter " +
                    "convolution algorithm. 'fastest' used instead")
                fallthrough
            }

        case .fastest:
            let algs = try findBackwardFilterAlgorithms(x: x)
            bwdFilterAlgo = algs[0].algo
            bwdFilterWorkspaceSize = algs[0].memory
            
        case .noWorkspace:
            let algs = try findBackwardFilterAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory == 0 { algIndex = i; break }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find 'noWorkspace' backward filter " +
                    "convolution algorithm")
                throw DeviceError.initializeFailed
            }
            bwdFilterAlgo = algs[algIndex].algo
            bwdFilterWorkspaceSize = algs[algIndex].memory
            
        case .workspaceLimit:
            let algs = try findBackwardFilterAlgorithms(x: x)
            var algIndex = -1
            for i in 0..<algs.count {
                if algs[i].memory <= properties.backwardFilterWorkspaceLimit {
                    algIndex = i
                    break
                }
            }
            
            guard algIndex >= 0 else {
                writeLog("failed to find suitable 'workspaceLimit' " +
                    "backward filter convolution algorithm")
                throw DeviceError.initializeFailed
            }
            bwdFilterAlgo = algs[algIndex].algo
            bwdFilterWorkspaceSize = algs[algIndex].memory
            
        default:
            // user explicitly specifies
            bwdFilterAlgo = properties.backwardFilterAlgorithm.cudnn
            
            // get the workspace size
            try cudaCheck(status: cudnnGetConvolutionBackwardFilterWorkspaceSize(
                dataQueue.cudnn.handle,
                xTensorDescriptor.desc,
                yTensorDescriptor.desc,
                convolutionDescriptor.desc,
                filterDescriptor.desc,
                bwdFilterAlgo,
                &bwdFilterWorkspaceSize))
        }
        
        // allocate workspace
        if bwdFilterWorkspaceSize > 0 {
            bwdFilterWorkspace = try dataQueue.device
                .createArray(byteCount: bwdFilterWorkspaceSize,
                             heapIndex: 0, zero: false)
        }

        // report selection
        let filterAlg = ConvolutionBwdFilterAlgorithm(cudnn: bwdFilterAlgo)

        if willLog(level: .diagnostic) &&
            properties.backwardFilterAlgorithm != filterAlg
        {
            diagnostic("using backward filter algorithm: " +
                "\(filterAlg)  workspace size: \(bwdFilterWorkspaceSize)",
                categories: logCategories)
        }
    }
    
    //--------------------------------------------------------------------------
    // findForwardAlgorithms
    private func findForwardAlgorithms(x: T) throws ->
        [cudnnConvolutionFwdAlgoPerf_t]
    {
        // get the list of forward algorithms
        var returnedAlgoCount: Int32 = 0
        var results = [cudnnConvolutionFwdAlgoPerf_t](
            repeating: cudnnConvolutionFwdAlgoPerf_t(),
            count: ConvolutionFwdAlgorithm.allCases.count)

        try cudaCheck(status: cudnnFindConvolutionForwardAlgorithm(
            dataQueue.cudnn.handle,
            xTensorDescriptor.desc,
            filterDescriptor.desc,
            convolutionDescriptor.desc,
            yTensorDescriptor.desc,
            Int32(results.count),
            &returnedAlgoCount,
            &results))

        // report
        if willLog(level: .diagnostic) {
            diagnostic("", categories: logCategories)
            diagnostic("find forward algorithms",
                       categories: logCategories, trailing: "-")

            for item in results {
                let alg = ConvolutionFwdAlgorithm(cudnn: item.algo)
                let det = item.determinism == CUDNN_DETERMINISTIC ?
                    "deterministic" : "non-deterministic"
                diagnostic("Algorithm: \(alg)  time: \(item.time) " +
                    "required memory: \(item.memory)  \(det)",
                    categories: logCategories)
            }
        }
        
        results.removeLast(results.count - Int(returnedAlgoCount))
        return results
    }
    
    //--------------------------------------------------------------------------
    // findBackwardDataAlgorithms
    private func findBackwardDataAlgorithms(x: T) throws ->
        [cudnnConvolutionBwdDataAlgoPerf_t]
    {
        // get the list of forward algorithms
        var returnedAlgoCount: Int32 = 0
        var results = [cudnnConvolutionBwdDataAlgoPerf_t](
            repeating: cudnnConvolutionBwdDataAlgoPerf_t(),
            count: ConvolutionBwdDataAlgorithm.allCases.count)

        try cudaCheck(status: cudnnFindConvolutionBackwardDataAlgorithm(
            dataQueue.cudnn.handle,
            filterDescriptor.desc,
            yTensorDescriptor.desc,
            convolutionDescriptor.desc,
            xTensorDescriptor.desc,
            Int32(results.count),
            &returnedAlgoCount,
            &results))
        
        if willLog(level: .diagnostic) {
            diagnostic("", categories: logCategories)
            diagnostic("find backward data algorithms",
                       categories: logCategories, trailing: "-")
            
            for item in results {
                let alg = ConvolutionBwdDataAlgorithm(cudnn: item.algo)
                let det = item.determinism == CUDNN_DETERMINISTIC ?
                    "deterministic" : "non-deterministic"
                diagnostic("Algorithm: \(alg)  time: \(item.time) " +
                    "required memory: \(item.memory)  \(det)",
                    categories: logCategories)
            }
        }
        
        results.removeLast(results.count - Int(returnedAlgoCount))
        return results
    }
    
    //--------------------------------------------------------------------------
    // findBackwardFilterAlgorithms
    private func findBackwardFilterAlgorithms(x: T) throws ->
        [cudnnConvolutionBwdFilterAlgoPerf_t]
    {
        // get the list of forward algorithms
        var returnedAlgoCount: Int32 = 0
        var results = [cudnnConvolutionBwdFilterAlgoPerf_t](
            repeating: cudnnConvolutionBwdFilterAlgoPerf_t(),
            count: ConvolutionBwdFilterAlgorithm.allCases.count)
        
        try cudaCheck(status: cudnnFindConvolutionBackwardFilterAlgorithm(
            dataQueue.cudnn.handle,
            xTensorDescriptor.desc,
            yTensorDescriptor.desc,
            convolutionDescriptor.desc,
            filterDescriptor.desc,
            Int32(results.count),
            &returnedAlgoCount,
            &results))
        
        if willLog(level: .diagnostic) {
            diagnostic("", categories: logCategories)
            diagnostic("find backward filter algorithms",
                       categories: logCategories, trailing: "-")
            
            for item in results {
                let alg = ConvolutionBwdFilterAlgorithm(cudnn: item.algo)
                let det = item.determinism == CUDNN_DETERMINISTIC ?
                    "deterministic" : "non-deterministic"
                diagnostic("Algorithm: \(alg)  time: \(item.time) " +
                    "required memory: \(item.memory)  \(det)",
                    categories: logCategories)
            }
        }
        
        results.removeLast(results.count - Int(returnedAlgoCount))
        return results
    }
}

//==============================================================================
// ConvolutionFwdAlgorithm
extension cudnnConvolutionFwdAlgo_t : Hashable {}

extension ConvolutionFwdAlgorithm {
    public var cudnn: cudnnConvolutionFwdAlgo_t {
        get {
            let algs: [ConvolutionFwdAlgorithm: cudnnConvolutionFwdAlgo_t] = [
                .implicitGEMM: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                .implicitPrecompGEMM: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                .gemm: CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                .direct: CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                .fft: CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                .fftTiling: CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                .winograd: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                .winogradNonFused: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            ]
            return algs[self]!
        }
    }
    
    public init(cudnn: cudnnConvolutionFwdAlgo_t) {
        let algs: [cudnnConvolutionFwdAlgo_t: ConvolutionFwdAlgorithm] = [
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: .implicitGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: .implicitPrecompGEMM,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM: .gemm,
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: .direct,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT: .fft,
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: .fftTiling,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: .winograd,
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: .winogradNonFused,
        ]
        self = algs[cudnn]!
    }
}

//==============================================================================
// ConvolutionBwdDataAlgorithm
extension cudnnConvolutionBwdDataAlgo_t : Hashable {}

extension ConvolutionBwdDataAlgorithm {
    public var cudnn: cudnnConvolutionBwdDataAlgo_t {
        get {
            let algs: [ConvolutionBwdDataAlgorithm: cudnnConvolutionBwdDataAlgo_t] = [
                .algo0: CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                .algo1: CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                .fft: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
                .fftTiling: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
                .winograd: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
                .winogradNonFused: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
            ]
            return algs[self]!
        }
    }
    
    public init(cudnn: cudnnConvolutionBwdDataAlgo_t) {
        let algs: [cudnnConvolutionBwdDataAlgo_t: ConvolutionBwdDataAlgorithm] = [
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: .algo0,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: .algo1,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: .fft,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: .fftTiling,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: .winograd,
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: .winogradNonFused
        ]
        self = algs[cudnn]!
    }
}

//==============================================================================
// ConvolutionBwdFilterAlgorithm
extension cudnnConvolutionBwdFilterAlgo_t : Hashable {}

extension ConvolutionBwdFilterAlgorithm {
    public var cudnn: cudnnConvolutionBwdFilterAlgo_t {
        get {
            let algs: [ConvolutionBwdFilterAlgorithm: cudnnConvolutionBwdFilterAlgo_t] = [
                .algo0: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                .algo1: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                .algo3: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
                .fft: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
                .winograd: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
                .winogradNonFused: CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
            ]
            return algs[self]!
        }
    }

    public init(cudnn: cudnnConvolutionBwdFilterAlgo_t) {
        let algs: [cudnnConvolutionBwdFilterAlgo_t: ConvolutionBwdFilterAlgorithm] = [
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0: .algo0,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1: .algo1,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3: .algo3,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT: .fft,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD: .winograd,
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: .winogradNonFused,
        ]
        self = algs[cudnn]!
    }
}

//==============================================================================
// ConvolutionDescriptor
public final class ConvolutionDescriptor : ObjectTracking {
    // properties
    public let trackingId: Int
    public let desc: cudnnConvolutionDescriptor_t

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
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

        trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self)
    }

    //--------------------------------------------------------------------------
    // deinit
    @inlinable
    deinit {
        try! cudaCheck(status: cudnnDestroyConvolutionDescriptor(desc))
        ObjectTracker.global.remove(trackingId: trackingId)
    }
}

//==============================================================================
// ConvolutionMode
extension ConvolutionMode {
    public var cudnn: cudnnConvolutionMode_t {
        get {
            switch self {
            case .convolution: return CUDNN_CONVOLUTION
            case .crossCorrelation: return CUDNN_CROSS_CORRELATION
            }
        }
    }
}

