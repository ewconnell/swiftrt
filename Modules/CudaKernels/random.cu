//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
#include "include/kernelHelpers.h"

#include <curand_kernel.h>

#ifdef __JETBRAINS_IDE__
#include "../../../../../../usr/local/cuda/include/driver_types.h"
#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../../usr/local/cuda/include/curand_kernel.h"
#include "../../../../../../usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h"
#include "../../../../../../usr/include/assert.h"
#include "../../../../../../usr/local/cuda/include/cuda_fp16.h"

#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; };
extern struct __cuda_fake_struct gridDim;
extern struct __cuda_fake_struct threadIdx;
extern struct __cuda_fake_struct blockIdx;
#endif

//------------------------------------------------------------------------------
// cudaCreateRandomGeneratorState
__global__ void initRandom_kernel(curandState *states,
								  unsigned long long seed, size_t count)
{
	KERNEL_LOOP(i, count) {
		curand_init(seed, i, 0, &states[i]);
	}
}

cudaError_t cudaCreateRandomGeneratorState(void** generatorState,
										   unsigned long long seed,
										   size_t count, cudaStream_t stream)
{
	assert(generatorState != NULL);
	CudaKernelPreCheck(stream);

	// allocate space on the GPU for the random states
	cudaError_t status = cudaMalloc(generatorState, count * sizeof(curandState_t));
	if(status != cudaSuccess) return status;

	initRandom_kernel<<<BLOCK_COUNT(count), THREADS_PER_BLOCK, 0, stream>>>
			((curandState *)*generatorState, seed, count);

	return CudaKernelPostCheck(stream);
}

//------------------------------------------------------------------------------
// cudaFillUniform
__global__ void fillUniform_kernel(curandState *state,
								   const cudaShape_t shape, float* data)
{
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = curand_uniform(&state[i]);
	}
}

__global__ void fillUniform_kernel(curandState *state,
								   const cudaShape_t shape, double* data)
{
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = curand_uniform_double(&state[i]);
	}
}

// cudaFillUniform
cudaError_t cudaFillUniform(const cudaShape_t shape, void *data,
							void* generatorState, cudaStream_t stream)
{
	assert(generatorState != NULL && data != NULL);
	CudaKernelPreCheck(stream);

	// require flat
	unsigned numBlocks  = BLOCK_COUNT(shape.elementCount);
	unsigned numThreads = THREADS_PER_BLOCK;

	switch(shape.dataType) {
		case CUDA_R_32F:
			fillUniform_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, shape, (float*)data);
			break;

		case CUDA_R_64F:
			fillUniform_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, shape, (double*)data);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}

//------------------------------------------------------------------------------
// cudaFillGaussian
// http://stats.stackexchange.com/questions/46429/transform-data-to-desired-mean-and-standard-deviation
//
__global__ void fillGaussian_kernel(curandState *state, float mean, float std,
									const cudaShape_t shape, float* data)
{
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = mean + std * curand_normal(&state[i]);
	}
}

__global__ void fillGaussian_kernel(curandState *state, double mean, double std,
									const cudaShape_t shape, double* data)
{
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = mean + std * curand_normal_double(&state[i]);
	}
}

// cudaFillGaussian
cudaError_t cudaFillGaussian(const cudaShape_t shape, void *data,
							 double mean, double std,
							 void* generatorState, cudaStream_t stream)
{
	assert(generatorState != NULL && data != NULL);
	CudaKernelPreCheck(stream);

	// require flat
	unsigned numBlocks  = BLOCK_COUNT(shape.elementCount);
	unsigned numThreads = THREADS_PER_BLOCK;

	switch(shape.dataType) {
		case CUDA_R_32F:
			fillGaussian_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, (float)mean, (float)std, shape, (float*)data);
			break;

		case CUDA_R_64F:
			fillGaussian_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, mean, std, shape, (double*)data);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}

//------------------------------------------------------------------------------
// cudaFillXavier
__global__ void fillXavier_kernel(curandState *state, float range,
								  const cudaShape_t shape, half* data) {
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = __float2half((curand_uniform(&state[i]) - 0.5f) * range);
	}
}

__global__ void fillXavier_kernel(curandState *state, float range,
								  const cudaShape_t shape, float* data) {
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = (curand_uniform(&state[i]) - 0.5f) * range;
	}
}

__global__ void fillXavier_kernel(curandState *state, double range,
								  const cudaShape_t shape, double* data) {
	KERNEL_LOOP(i, shape.elementCount) {
		data[i] = (curand_uniform_double(&state[i]) - 0.5) * range;
	}
}

// cudaFillXavier
cudaError_t cudaFillXavier(const cudaShape_t shape, void *data,
						   double varianceNorm, void* generatorState,
						   cudaStream_t stream)
{
	assert(generatorState != NULL && data != NULL);
	// require flat for now
	unsigned numBlocks  = BLOCK_COUNT(shape.elementCount);
	unsigned numThreads = THREADS_PER_BLOCK;

	double range = sqrt(3.0 / varianceNorm) * 2;

	CudaKernelPreCheck(stream);

	switch(shape.dataType) {
		case CUDA_R_16F:
			fillXavier_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, (float)range, shape, (half*)data);
			break;

		case CUDA_R_32F:
			fillXavier_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, (float)range, shape, (float*)data);
			break;

		case CUDA_R_64F:
			fillXavier_kernel<<<numBlocks, numThreads, 0, stream>>>
					((curandState *)generatorState, range, shape, (double*)data);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}

