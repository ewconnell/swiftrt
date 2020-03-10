//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"

#ifdef __JETBRAINS_IDE__
#include "../../../../../../usr/local/cuda/include/driver_types.h"
#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../../usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h"
#include "../../../../../../usr/include/assert.h"
#include "../../../../../../usr/include/stdint.h"

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
// device kernel
template <typename T, typename U>
__global__ void cudaCompareEqual_kernel1(
	unsigned elementCount,
	const T* aData, unsigned aStride,
	const T* bData, unsigned bStride,
	U* resultData,  unsigned rStride)
{
	CUDA_KERNEL_LOOP(i, elementCount) {
		resultData[i * rStride] = aData[i * aStride] == bData[i * bStride] ? (U)1 : (U)0;
	}
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t cudaCompareEqual(
		size_t elementCount,
		enum cudaDataType_t abDataType,
		const void* aData, size_t aStride,
		const void* bData, size_t bStride,
		enum cudaDataType_t resultDataType,
		void* resultData, size_t rStride,
		cudaStream_t stream) {
// validate
	CudaKernelPreCheck(stream);

	unsigned numBlocks = CUDA_NUM_BLOCKS(elementCount);
	unsigned numThreads = CUDA_NUM_THREADS;
	switch(abDataType) {
		case CUDA_R_32I:
			switch(resultDataType) {
				case CUDA_R_32F:
				cudaCompareEqual_kernel1 << < numBlocks, numThreads, 0, stream >> >
					(elementCount,
					(int32_t *) aData, aStride,
					(int32_t *) bData, bStride,
					(float *) resultData, rStride);
					break;
				default:
					// not implemented
					assert(false);
			}
			break;

		case CUDA_R_32F:
			cudaCompareEqual_kernel1<<<numBlocks, numThreads, 0, stream>>>
					(elementCount,
							(float*)aData, aStride,
							(float*)bData, bStride,
							(float*)resultData, rStride);
			break;

		case CUDA_R_64F:
			cudaCompareEqual_kernel1<<<numBlocks, numThreads, 0, stream>>>
					(elementCount,
							(double*)aData, aStride,
							(double*)bData, bStride,
							(double*)resultData, rStride);
			break;

		default: assert(false);
	};
	return CudaKernelPostCheck(stream);
}
