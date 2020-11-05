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

import SwiftRTCuda

//==============================================================================
// CudaQueue `convolution` implementation
extension CudaQueue {
  public func convolution<Shape, Element, FilterElement>(
    activation: ActivationType,
    strides: Shape,
    padding: Padding,
    dilations: Shape,
    properties: ConvolutionProperties,
    deviceId: Int,
    filterBiasBackpropQueueIndex: Int
  ) -> DeviceConvolution<Shape, Element, FilterElement>
  where
    Shape: TensorShape,
    Element: StorageElement & Numeric,
    FilterElement: StorageElement & Numeric
  {
    if useGpu {
      return CudaConvolution<Shape, Element, FilterElement>(
        activation: activation,
        strides: strides,
        padding: padding,
        dilations: dilations,
        properties: properties,
        deviceId: deviceId,
        filterBiasBackpropQueueIndex: filterBiasBackpropQueueIndex)
    } else {
      return CpuConvolution<Shape, Element, FilterElement>(
        activation: activation,
        strides: strides,
        padding: padding,
        dilations: dilations,
        properties: properties,
        deviceId: 0,
        filterBiasBackpropQueueIndex: filterBiasBackpropQueueIndex)
    }
  }
}

//==============================================================================
// CudaConvolution
public final class CudaConvolution<Shape, Element, FilterElement>:
  DeviceConvolution<Shape, Element, FilterElement>
where
  Shape: TensorShape,
  Element: StorageElement & Numeric,
  FilterElement: StorageElement & Numeric
{
  // constants
  public var zero = Element.zero
  public var one = Element.one

  // descriptors
  public let activationDescriptor: ActivationDescriptor

  // retained tensors
  public var y: Data!

  // queues
  public let dataQueue: CudaQueue
  public let filterBiasBackQueue: CudaQueue

  public var convolutionDescriptor: ConvolutionDescriptor<Shape>!
  public var xTensorDescriptor: TensorDescriptor<Shape, Element>!
  public var yTensorDescriptor: TensorDescriptor<Shape, Element>!
  public var filterDescriptor: FilterDescriptor!
  public var biasTensorDescriptor: TensorDescriptor<Shape1, FilterElement>!

  // forward
  public var fwdAlgo: cudnnConvolutionFwdAlgo_t
  public var fwdWorkspaceSize = 0
  public var fwdWorkspace: DeviceMemory?

  // backward data
  public var bwdDataAlgo: cudnnConvolutionBwdDataAlgo_t
  public var bwdDataWorkspaceSize = 0
  public var bwdDataWorkspace: DeviceMemory?

  // backward filter
  public var bwdFilterAlgo: cudnnConvolutionBwdFilterAlgo_t
  public var bwdFilterWorkspaceSize = 0
  public var bwdFilterWorkspace: DeviceMemory?
  public let logCategories: LogCategories = [.setup]

  //--------------------------------------------------------------------------
  // initializer
  @inlinable public override init(
    activation: ActivationType,
    strides: Shape,
    padding: Padding,
    dilations: Shape,
    properties: ConvolutionProperties,
    deviceId: Int,
    filterBiasBackpropQueueIndex: Int
  ) {
    // these values are just init place holders and will be set
    // correctly during `setupForward`
    self.fwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    self.bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
    self.bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0

    //----------------------------------
    // queues
    // TODO: change this when devices have fixed collections of queues
    // initialization can create workspaces on the devices
    // associated with the queues, so we hold on to them
    let defaultQueue = currentQueue
    self.dataQueue = defaultQueue
    self.filterBiasBackQueue = defaultQueue

    //----------------------------------
    // create activation descriptor
    self.activationDescriptor = ActivationDescriptor(
      mode: activation,
      nan: properties.activationNan,
      reluCeiling: properties.activationReluCeiling)

    //----------------------------------
    // stored common properties
    super.init(
      activation: activation,
      strides: strides,
      padding: padding,
      dilations: dilations,
      properties: properties,
      deviceId: deviceId,
      filterBiasBackpropQueueIndex: filterBiasBackpropQueueIndex
    )
  }

  //--------------------------------------------------------------------------
  // forward
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
  @inlinable public override func forward(
    x: Data,
    filter: Filter,
    bias: Bias,
    mode: EvaluationMode
  ) -> Data {
    // setup any time the input shape changes
    if x.shape != inputShape {
      setupForward(x, filter, bias, mode)
    }

    cudaCheck(
      cudnnConvolutionBiasActivationForward(
        dataQueue.cudnn.handle,
        // alpha1
        &one,
        // x
        xTensorDescriptor.desc,
        x.deviceRead(using: dataQueue),
        // filter weights
        filterDescriptor.desc,
        filter.deviceRead(using: dataQueue),
        // convDesc
        convolutionDescriptor.desc,
        // algo
        fwdAlgo,
        // workspace device array
        fwdWorkspace?.mutablePointer,
        // workspace size in bytes
        fwdWorkspaceSize,
        // alpha2
        &zero,
        // z used for activation (TODO: inplace on y?? find out what's right)
        yTensorDescriptor.desc,
        y.deviceRead(using: dataQueue),
        // bias
        biasTensorDescriptor.desc,
        bias.deviceRead(using: dataQueue),
        // activation
        activationDescriptor.desc,
        // y
        yTensorDescriptor.desc,
        y.deviceReadWrite(using: dataQueue)))

    return y
  }

  //--------------------------------------------------------------------------
  // backward
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
  @inlinable public override func backward(
    y: Data,
    yDiff: Data,
    filter: Filter,
    filterDiff: inout Filter,
    bias: Bias,
    biasDiff: inout Bias,
    x: Data,
    xDiff: inout Data,
    mode: EvaluationMode
  ) {
    // let pone = UnsafeRawPointer(&one) { $0 }
    // data
    cudaCheck(
      cudnnConvolutionBackwardData(
        dataQueue.cudnn.handle,
        // alpha
        &one,
        // filter
        filterDescriptor.desc,
        filter.deviceRead(using: dataQueue),
        // yDiff
        yTensorDescriptor.desc,
        yDiff.deviceRead(using: dataQueue),
        // conv
        convolutionDescriptor.desc,
        // algo
        bwdDataAlgo,
        // workspace
        bwdDataWorkspace?.mutablePointer,
        bwdDataWorkspaceSize,
        // beta
        &zero,
        // xDiff
        xTensorDescriptor.desc,
        xDiff.deviceReadWrite(using: dataQueue)))

    // filter
    cudaCheck(
      cudnnConvolutionBackwardFilter(
        filterBiasBackQueue.cudnn.handle,
        // alpha
        &one,
        // x
        xTensorDescriptor.desc,
        x.deviceRead(using: dataQueue),
        // yDiff
        yTensorDescriptor.desc,
        yDiff.deviceRead(using: dataQueue),
        // conv
        convolutionDescriptor.desc,
        // algo
        bwdFilterAlgo,
        // workspace
        bwdFilterWorkspace?.mutablePointer,
        bwdFilterWorkspaceSize,
        // beta
        &zero,
        // filterDiff
        filterDescriptor.desc,
        filterDiff.deviceReadWrite(using: dataQueue)))

    // bias
    cudaCheck(
      cudnnConvolutionBackwardBias(
        filterBiasBackQueue.cudnn.handle,
        // alpha
        &one,
        // yDiff
        yTensorDescriptor.desc,
        yDiff.deviceRead(using: dataQueue),
        // beta
        &zero,
        //
        biasTensorDescriptor.desc,
        biasDiff.deviceReadWrite(using: dataQueue)))
  }

  //--------------------------------------------------------------------------
  // setupForward
  @inlinable public func setupForward(
    _ x: Data,
    _ filter: Filter,
    _ bias: Bias,
    _ mode: EvaluationMode
  ) {
    xTensorDescriptor = TensorDescriptor(x)
    filterDescriptor = FilterDescriptor(filter)
    biasTensorDescriptor = TensorDescriptor(bias)

    //----------------------------------
    // create convolution descriptor
    let convolutionStorageElementType: cudnnDataType_t =
      Element.type == real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT

    let pad = (padding == .valid) ? Shape.zero : (filter.shape / 2)

    convolutionDescriptor = ConvolutionDescriptor(
      scalarType: convolutionStorageElementType,
      rank: Shape.rank - 2,
      padding: pad,
      strides: strides,
      dilations: dilations,
      mode: properties.mode)

    //----------------------------------
    // get the extents for the output
    var yExtent = [Int32](repeating: 0, count: Shape.rank)
    cudaCheck(
      cudnnGetConvolutionNdForwardOutputDim(
        convolutionDescriptor.desc,
        xTensorDescriptor.desc,
        filterDescriptor.desc,
        Int32(yExtent.count),
        &yExtent))

    //----------------------------------
    // return the shape of the output y and create a tensorDescriptor
    // with the same scalarType for y as x

    // TODO: let yShape = Shape(yExtent.map(Int.init))
    yTensorDescriptor = TensorDescriptor(x)
    selectForwardAlgorithm(x: x, properties: properties)

    if mode == .training {
      selectBackwardAlgorithm(x: x, properties: properties)
    }
  }

  //--------------------------------------------------------------------------
  // selectForwardAlgorithm
  @inlinable public func selectForwardAlgorithm(
    x: Data,
    properties: ConvolutionProperties
  ) {
    switch properties.forwardAlgorithm {
    case .deterministic:
      let algs = findForwardAlgorithms(x: x)
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
        writeLog(
          "failed to find 'deterministic' forward "
            + "convolution algorithm. 'fastest' used instead")
        fallthrough
      }

    case .fastest:
      let algs = findForwardAlgorithms(x: x)
      fwdAlgo = algs[0].algo
      fwdWorkspaceSize = algs[0].memory

    case .noWorkspace:
      let algs = findForwardAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory == 0 {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog("failed to find 'noWorkspace' forward " + "convolution algorithm")
        fatalError("convolution initialization failed")
      }
      fwdAlgo = algs[algIndex].algo
      fwdWorkspaceSize = algs[algIndex].memory

    case .workspaceLimit:
      let algs = findForwardAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory <= properties.forwardWorkspaceLimit {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog("failed to find suitable 'workspaceLimit' " + "forward convolution algorithm")
        fatalError("convolution initialization failed")
      }
      fwdAlgo = algs[algIndex].algo
      fwdWorkspaceSize = algs[algIndex].memory

    default:
      // user explicitly specifies
      fwdAlgo = properties.forwardAlgorithm.cudnn

      // get the workspace size
      cudaCheck(
        cudnnGetConvolutionForwardWorkspaceSize(
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
      fwdWorkspace = dataQueue.allocate(fwdWorkspaceSize)
    }

    // report selection
    let alg = ConvolutionFwdAlgorithm(cudnn: fwdAlgo)

    if willLog(level: .diagnostic) && properties.forwardAlgorithm != alg {
      diagnostic(
        .setup, "using forward algorithm: " + "\(alg)  workspace size: \(fwdWorkspaceSize)",
        categories: logCategories)
    }
  }

  //--------------------------------------------------------------------------
  // selectBackwardAlgorithm
  @inlinable public func selectBackwardAlgorithm(
    x: Data,
    properties: ConvolutionProperties
  ) {
    switch properties.backwardDataAlgorithm {
    case .deterministic:
      let algs = findBackwardDataAlgorithms(x: x)
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
        writeLog(
          "failed to find 'deterministic' backward data "
            + "convolution algorithm. 'fastest' used instead")
        fallthrough
      }

    case .fastest:
      let algs = findBackwardDataAlgorithms(x: x)
      bwdDataAlgo = algs[0].algo
      bwdDataWorkspaceSize = algs[0].memory

    case .noWorkspace:
      let algs = findBackwardDataAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory == 0 {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog("failed to find 'noWorkspace' backward data " + "convolution algorithm")
        fatalError("convolution initialization failed")
      }
      bwdDataAlgo = algs[algIndex].algo
      bwdDataWorkspaceSize = algs[algIndex].memory

    case .workspaceLimit:
      let algs = findBackwardDataAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory <= properties.backwardDataWorkspaceLimit {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog(
          "failed to find suitable 'workspaceLimit' " + "backward data convolution algorithm")
        fatalError("convolution initialization failed")
      }
      bwdDataAlgo = algs[algIndex].algo
      bwdDataWorkspaceSize = algs[algIndex].memory

    default:
      // user explicitly specifies
      bwdDataAlgo = properties.backwardDataAlgorithm.cudnn

      // get the workspace size
      cudaCheck(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
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
      bwdDataWorkspace = dataQueue.allocate(bwdDataWorkspaceSize)
    }

    // report selection
    let dataAlg = ConvolutionBwdDataAlgorithm(cudnn: bwdDataAlgo)

    if willLog(level: .diagnostic) && properties.backwardDataAlgorithm != dataAlg {
      diagnostic(
        .setup,
        "using backward data algorithm: " + "\(dataAlg)  workspace size: \(bwdDataWorkspaceSize)",
        categories: logCategories)
    }

    //----------------------------------
    // choose best backward filter algorithm
    switch properties.backwardFilterAlgorithm {
    case .deterministic:
      let algs = findBackwardFilterAlgorithms(x: x)
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
        writeLog(
          "failed to find 'deterministic' backward filter "
            + "convolution algorithm. 'fastest' used instead")
        fallthrough
      }

    case .fastest:
      let algs = findBackwardFilterAlgorithms(x: x)
      bwdFilterAlgo = algs[0].algo
      bwdFilterWorkspaceSize = algs[0].memory

    case .noWorkspace:
      let algs = findBackwardFilterAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory == 0 {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog("failed to find 'noWorkspace' backward filter " + "convolution algorithm")
        fatalError("convolution initialization failed")
      }
      bwdFilterAlgo = algs[algIndex].algo
      bwdFilterWorkspaceSize = algs[algIndex].memory

    case .workspaceLimit:
      let algs = findBackwardFilterAlgorithms(x: x)
      var algIndex = -1
      for i in 0..<algs.count {
        if algs[i].memory <= properties.backwardFilterWorkspaceLimit {
          algIndex = i
          break
        }
      }

      if algIndex == -1 {
        writeLog(
          "failed to find suitable 'workspaceLimit' " + "backward filter convolution algorithm")
        fatalError("convolution initialization failed")
      }
      bwdFilterAlgo = algs[algIndex].algo
      bwdFilterWorkspaceSize = algs[algIndex].memory

    default:
      // user explicitly specifies
      bwdFilterAlgo = properties.backwardFilterAlgorithm.cudnn

      // get the workspace size
      cudaCheck(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
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
      bwdFilterWorkspace = dataQueue.allocate(bwdFilterWorkspaceSize)
    }

    // report selection
    let filterAlg = ConvolutionBwdFilterAlgorithm(cudnn: bwdFilterAlgo)

    if willLog(level: .diagnostic) && properties.backwardFilterAlgorithm != filterAlg {
      diagnostic(
        .setup,
        "using backward filter algorithm: "
          + "\(filterAlg)  workspace size: \(bwdFilterWorkspaceSize)",
        categories: logCategories)
    }
  }

  //--------------------------------------------------------------------------
  // findForwardAlgorithms
  @inlinable public func findForwardAlgorithms(
    x: Data
  ) -> [cudnnConvolutionFwdAlgoPerf_t] {
    // get the list of forward algorithms
    var returnedAlgoCount: Int32 = 0
    var results = [cudnnConvolutionFwdAlgoPerf_t](
      repeating: cudnnConvolutionFwdAlgoPerf_t(),
      count: ConvolutionFwdAlgorithm.allCases.count)

    cudaCheck(
      cudnnFindConvolutionForwardAlgorithm(
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
      diagnostic(.setup, "", categories: logCategories)
      diagnostic(
        .setup, "find forward algorithms",
        categories: logCategories, trailing: "-")

      for item in results {
        let alg = ConvolutionFwdAlgorithm(cudnn: item.algo)
        let det = item.determinism == CUDNN_DETERMINISTIC ? "deterministic" : "non-deterministic"
        diagnostic(
          .setup,
          "Algorithm: \(alg)  time: \(item.time) " + "required memory: \(item.memory)  \(det)",
          categories: logCategories)
      }
    }

    results.removeLast(results.count - Int(returnedAlgoCount))
    return results
  }

  //--------------------------------------------------------------------------
  // findBackwardDataAlgorithms
  @inlinable public func findBackwardDataAlgorithms(
    x: Data
  ) -> [cudnnConvolutionBwdDataAlgoPerf_t] {
    // get the list of forward algorithms
    var returnedAlgoCount: Int32 = 0
    var results = [cudnnConvolutionBwdDataAlgoPerf_t](
      repeating: cudnnConvolutionBwdDataAlgoPerf_t(),
      count: ConvolutionBwdDataAlgorithm.allCases.count)

    cudaCheck(
      cudnnFindConvolutionBackwardDataAlgorithm(
        dataQueue.cudnn.handle,
        filterDescriptor.desc,
        yTensorDescriptor.desc,
        convolutionDescriptor.desc,
        xTensorDescriptor.desc,
        Int32(results.count),
        &returnedAlgoCount,
        &results))

    if willLog(level: .diagnostic) {
      diagnostic(.setup, "", categories: logCategories)
      diagnostic(
        .setup, "find backward data algorithms",
        categories: logCategories, trailing: "-")

      for item in results {
        let alg = ConvolutionBwdDataAlgorithm(cudnn: item.algo)
        let det = item.determinism == CUDNN_DETERMINISTIC ? "deterministic" : "non-deterministic"
        diagnostic(
          .setup,
          "Algorithm: \(alg)  time: \(item.time) " + "required memory: \(item.memory)  \(det)",
          categories: logCategories)
      }
    }

    results.removeLast(results.count - Int(returnedAlgoCount))
    return results
  }

  //--------------------------------------------------------------------------
  // findBackwardFilterAlgorithms
  @inlinable public func findBackwardFilterAlgorithms(
    x: Data
  ) -> [cudnnConvolutionBwdFilterAlgoPerf_t] {
    // get the list of forward algorithms
    var returnedAlgoCount: Int32 = 0
    var results = [cudnnConvolutionBwdFilterAlgoPerf_t](
      repeating: cudnnConvolutionBwdFilterAlgoPerf_t(),
      count: ConvolutionBwdFilterAlgorithm.allCases.count)

    cudaCheck(
      cudnnFindConvolutionBackwardFilterAlgorithm(
        dataQueue.cudnn.handle,
        xTensorDescriptor.desc,
        yTensorDescriptor.desc,
        convolutionDescriptor.desc,
        filterDescriptor.desc,
        Int32(results.count),
        &returnedAlgoCount,
        &results))

    if willLog(level: .diagnostic) {
      diagnostic(.setup, "", categories: logCategories)
      diagnostic(
        .setup, "find backward filter algorithms",
        categories: logCategories, trailing: "-")

      for item in results {
        let alg = ConvolutionBwdFilterAlgorithm(cudnn: item.algo)
        let det = item.determinism == CUDNN_DETERMINISTIC ? "deterministic" : "non-deterministic"
        diagnostic(
          .setup,
          "Algorithm: \(alg)  time: \(item.time) " + "required memory: \(item.memory)  \(det)",
          categories: logCategories)
      }
    }

    results.removeLast(results.count - Int(returnedAlgoCount))
    return results
  }
}

//==============================================================================
// ConvolutionFwdAlgorithm
extension cudnnConvolutionFwdAlgo_t: Hashable {}

extension ConvolutionFwdAlgorithm {
  public var cudnn: cudnnConvolutionFwdAlgo_t {
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
extension cudnnConvolutionBwdDataAlgo_t: Hashable {}

extension ConvolutionBwdDataAlgorithm {
  public var cudnn: cudnnConvolutionBwdDataAlgo_t {
    let algs: [ConvolutionBwdDataAlgorithm: cudnnConvolutionBwdDataAlgo_t] = [
      .algo0: CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
      .algo1: CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
      .fft: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
      .fftTiling: CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
      .winograd: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
      .winogradNonFused: CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
    ]
    return algs[self]!
  }

  public init(cudnn: cudnnConvolutionBwdDataAlgo_t) {
    let algs: [cudnnConvolutionBwdDataAlgo_t: ConvolutionBwdDataAlgorithm] = [
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: .algo0,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: .algo1,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: .fft,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: .fftTiling,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: .winograd,
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: .winogradNonFused,
    ]
    self = algs[cudnn]!
  }
}

//==============================================================================
// ConvolutionBwdFilterAlgorithm
extension cudnnConvolutionBwdFilterAlgo_t: Hashable {}

extension ConvolutionBwdFilterAlgorithm {
  public var cudnn: cudnnConvolutionBwdFilterAlgo_t {
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
public final class ConvolutionDescriptor<Shape: TensorShape> {
  // properties
  public let desc: cudnnConvolutionDescriptor_t

  // initializers
  @inlinable public init(
    scalarType: cudnnDataType_t,
    rank: Int,
    padding: Shape,
    strides: Shape,
    dilations: Shape,
    mode: ConvolutionMode
  ) {
    // create the descriptor
    var temp: cudnnConvolutionDescriptor_t?
    cudaCheck(cudnnCreateConvolutionDescriptor(&temp))
    desc = temp!

    // initialize
    cudaCheck(
      cudnnSetConvolutionNdDescriptor(
        desc,
        Int32(rank),
        padding.asInt32,
        strides.asInt32,
        dilations.asInt32,
        mode.cudnn,
        scalarType))
  }

  @inlinable deinit {
    cudaCheck(cudnnDestroyConvolutionDescriptor(desc))
  }
}

//==============================================================================
// ConvolutionMode
extension ConvolutionMode {
  public var cudnn: cudnnConvolutionMode_t {
    switch self {
    case .convolution: return CUDNN_CONVOLUTION
    case .crossCorrelation: return CUDNN_CROSS_CORRELATION
    }
  }
}
