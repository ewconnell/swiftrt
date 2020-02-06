//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
#include <string.h>

//------------------------------------------------------------------------------
// cudaInitCudaShape
void cudaInitCudaShape(cudaShape_t *shape,
					   enum cudaDataType_t dataType,
					   cudnnTensorFormat_t format,
					   size_t extentCount,
					   const size_t *extent,
					   const size_t *stride,
					   size_t elementCount) {
	// clear the struct
	memset(shape, 0, sizeof(cudaShape_t));
	shape->dataType = dataType;
	shape->format = format;
	shape->extentCount = (unsigned)extentCount;
	shape->elementCount = (unsigned)elementCount;
	for(int i = 0; i < extentCount; ++i) {
		shape->extent[i] = (unsigned)extent[i];
		shape->stride[i] = (unsigned)stride[i];
	}
}
