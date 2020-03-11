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
#ifndef CudaKernels_h
#define CudaKernels_h

#include <driver_types.h>
#include <cuda_fp16.h>
#include <library_types.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <cudnn.h>

//==============================================================================
// make visible to Swift as C API
#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Cuda helpers

#define CUDA_KERNEL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)

#define CUDA_KERNEL_LOOP(i, n) \
  for (unsigned i = CUDA_KERNEL_INDEX; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// use 512 threads per block
const unsigned CUDA_NUM_THREADS = 1024;

// number of blocks for threads.
inline unsigned CUDA_NUM_BLOCKS(unsigned N) {
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

//==============================================================================
// launch error detection
inline void CudaKernelPreCheck(cudaStream_t stream) {
#ifdef DEBUG
	// reset error variable to cudaSuccess
	cudaGetLastError();
#endif
}

inline cudaError_t CudaKernelPostCheck(cudaStream_t stream)
{
#ifdef DEBUG
	cudaStreamSynchronize(stream);
	return cudaGetLastError();
#else
	return cudaSuccess;
#endif
}

//==============================================================================
// DataTypeSize
inline size_t DataTypeSize(enum cudaDataType_t type) {
	switch(type) {
		case CUDA_R_8U:  return (size_t)1;
		case CUDA_R_16F: return (size_t)2;
		case CUDA_R_32F: return (size_t)4;
		case CUDA_R_64F: return (size_t)8;
		default: assert(0);
	}
	return (size_t)0;
}

//==============================================================================
// DataShapeStruct
#define MAX_EXTENT_COUNT 5

typedef struct {
	enum cudaDataType_t dataType;
    cudnnTensorFormat_t format;
	unsigned extentCount;
	unsigned extent[MAX_EXTENT_COUNT];
	unsigned stride[MAX_EXTENT_COUNT];
	unsigned elementCount;
} cudaShape_t;

void cudaInitCudaShape(cudaShape_t *shape,
                       enum cudaDataType_t dataType,
                       cudnnTensorFormat_t format,
                       size_t extentCount,
                       const size_t *extent,
					   const size_t *stride,
					   size_t elementCount);

//==============================================================================
// kernels
cudaError_t cudaUpdateGradientWithMomentum(
        enum cudaDataType_t dataType, long count,
        void *weights, const void *gradient, double learningRate,
        void *history, double momentum,
        cudaStream_t stream);

cudaError_t cudaUpdateGradient(
        enum cudaDataType_t dataType, long count,
        void *weights, const void *gradient, double learningRate,
        cudaStream_t stream);

cudaError_t cudaCompareEqual(
		size_t elementCount,
		enum cudaDataType_t abDataType,
		const void* aData, size_t aStride,
		const void* bData, size_t bStride,
		enum cudaDataType_t resultDataType,
		void* resultData, size_t rStride,
		cudaStream_t stream);

cudaError_t cudaCreateRandomGeneratorState(void** generatorState,
										   unsigned long long seed,
										   size_t count, cudaStream_t stream);

cudaError_t cudaDelayStream(double seconds, int clockRate, cudaStream_t stream);

cudaError_t cudaDropoutForward(
	const cudaShape_t inShape, const void *inData,
	const cudaShape_t outShape, void *outData,
	double ratio, void *mask, void* generatorState, cudaStream_t stream);

cudaError_t cudaDropoutBackward(
	const cudaShape_t outShape, const void *outDiff,
	const cudaShape_t inShape, void *inDiff,
	const void *mask, cudaStream_t stream);


cudaError_t cudaExpandLabels(
	enum cudaDataType_t dataType,
	const void* labels, size_t labelStride,
	void *expanded, const size_t* expandedExtent, const size_t* expandedStrides,
	cudaStream_t stream);

cudaError_t cudaSoftmaxLabelGradient(
	long N,
	enum cudaDataType_t outDataType,
	const void* outData, long xCols, long rowStride,
	enum cudaDataType_t labelsDataType,
	const int* labels,
	double beta,
	enum cudaDataType_t inDiffDataType,
	void* inDiff,
	cudaStream_t stream);

cudaError_t cudaValidateRange(const cudaShape_t inShape, const void *inData,
							  const cudaShape_t outShape, void *outData,
							  cudaStream_t stream);

//-------------------------------------
cudaError_t cudaFillGaussian(const cudaShape_t shape, void *data,
							 double mean, double std,
							 void* generatorState, cudaStream_t stream);

cudaError_t cudaFillUniform(const cudaShape_t shape, void *data,
							void* generatorState, cudaStream_t stream);

cudaError_t cudaFillXavier(const cudaShape_t shape, void *data,
						   double varianceNorm, void* generatorState,
						   cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* CudaKernels_h */
