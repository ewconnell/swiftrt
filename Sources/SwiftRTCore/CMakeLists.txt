add_library(SwiftRTCore
  common/ElapsedTime.swift
  common/HashingUtilities.swift
  common/Helpers.swift
  common/Log.swift

  differentiable/ComparativeDerivatives.swift
  differentiable/DifferentialOperators.swift
  differentiable/FillDerivatives.swift
  differentiable/MathDerivatives.swift
  differentiable/ReductionDerivatives.swift
  differentiable/StdlibExtension.swift
  differentiable/TensorDerivatives.swift

  numpy/RankFunctions.swift
  numpy/array.swift
  numpy/arrayInit.swift
  numpy/eye.swift

  operators/AlgebraicField.swift
  operators/Comparative.swift
  operators/Fill.swift
  operators/Fractals.swift
  operators/Gather.swift
  operators/Infix.swift
  operators/Kernel.swift
  operators/Math.swift
  operators/Matmul.swift
  operators/Pool.swift
  operators/Reductions.swift

  platform/common/BFloat16.swift
  platform/common/ComplexExtensions.swift
  platform/common/DeviceQueue.swift
  platform/common/DiscreteStorage.swift
  platform/common/ExecutionPlanner.swift
  platform/common/FunctionProperties.swift
  platform/common/Platform.swift
  platform/common/StorageBuffer.swift
  platform/common/StorageElement.swift
  platform/common/pmap.swift

  platform/cpu/device/CpuDevice.swift
  platform/cpu/device/CpuEvent.swift
  platform/cpu/device/CpuPlatform.swift
  platform/cpu/device/CpuQueue.swift
  platform/cpu/device/CpuStorage.swift

  platform/cpu/functions/CpuConvolution.swift
  platform/cpu/functions/CpuFill.swift
  platform/cpu/functions/CpuMapOps.swift
  platform/cpu/functions/CpuMath.swift
  platform/cpu/functions/CpuMatmul.swift
  platform/cpu/functions/CpuPool.swift
  platform/cpu/functions/CpuReductions.swift

  random/RandomGenerators.swift
  random/RandomInit.swift
  random/RandomSeedState.swift

  tensor/Description.swift
  tensor/Ranges.swift
  tensor/Shape.swift
  tensor/Subscripts.swift
  tensor/SubscriptsUnbound.swift
  tensor/Tensor.swift
  tensor/TensorInit.swift
  tensor/VectorElement.swift)
if(SWIFTRT_ENABLE_CUDA)
  target_sources(SwiftRTCore PRIVATE
    platform/cuda/cublaslt/MatmulAlgorithm.swift
    platform/cuda/cublaslt/MatmulAlgorithmHeuristics.swift
    platform/cuda/cublaslt/MatmulAlgorithmSearch.swift
    platform/cuda/cublaslt/MatmulOperation.swift
    platform/cuda/cublaslt/MatmulPreferences.swift
    platform/cuda/cublaslt/MatrixLayout.swift
    platform/cuda/cublaslt/MatrixTransform.swift

    platform/cuda/device/CudaDevice.swift
    platform/cuda/device/CudaEvent.swift
    platform/cuda/device/CudaExtensions.swift
    platform/cuda/device/CudaPlatform.swift
    platform/cuda/device/CudaQueue.swift

    platform/cuda/functions/CudaCompare.swift
    platform/cuda/functions/CudaFill.swift
    platform/cuda/functions/CudaMath.swift
    platform/cuda/functions/CudaMathGen.swift
    platform/cuda/functions/CudaMatmul.swift
    platform/cuda/functions/CudaPool.swift
    platform/cuda/functions/CudaReductions.swift

    platform/cuda/layers/CudaActivation.swift
    platform/cuda/layers/CudaConvolution.swift)
  target_link_libraries(SwiftRTCore PUBLIC
    SwiftRTCuda)
  target_link_libraries(SwiftRTCore PUBLIC
    CCUDA)
endif()
set_target_properties(SwiftRTCore PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(SwiftRTCore PUBLIC
  $<$<AND:$<BOOL:dispatch_FOUND>,$<NOT:$<PLATFORM_ID:Darwin>>>:dispatch>
  $<$<AND:$<BOOL:Foundation_FOUND>,$<NOT:$<PLATFORM_ID:Darwin>>>:Foundation>
  Numerics)
