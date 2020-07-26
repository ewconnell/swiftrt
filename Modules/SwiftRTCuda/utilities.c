//
// Created by ed on 7/24/20.
//
#include <stdio.h>
#include <cuda_runtime.h>


int cudaTest() {
    return 42;
}

cudaError_t srtAdd(
    cudaDataType_t type,
    const void *a,
    const void *b,
    void *c,
    unsigned count,
    cudaStream_t stream
) {
    printf("I'm in srtAdd\n");
    return cudaSuccess;
}
