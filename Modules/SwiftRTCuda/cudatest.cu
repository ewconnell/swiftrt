// #include "precomp.cuh"

#include <chrono>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <iostream>
#include <vector>
using namespace std; 



#include "float16.cuh"
#include "bfloat16.cuh"
#include "complex.cuh"
#include "srt_traits.cuh"

#include "math_api.h"
#include "compare_api.h"

// #define TYPE  float
// #define OTYPE float
// #define DATYPE real32F
// #define DOTYPE real32F

// #define TYPE  float16
// #define OTYPE float16
// #define DATYPE real16F
// #define DOTYPE real16F

// #define TYPE  Complex<float16>
// #define OTYPE Complex<float16>
// #define DATYPE complex16F
// #define DOTYPE complex16F

#define TYPE  int8_t
#define OTYPE int8_t
#define DATYPE real8I
#define DOTYPE real8I

#define COUNT (1024 * 1024)


int main()
{
    // auto value = Complex<float>(1);
    // bool n = Complex<float>::isNormal(1);

    auto a = vector<TYPE>(COUNT);
    for (int i = 0; i < a.size(); ++i) a[i] = TYPE(float(i % 64));

    auto b = vector<TYPE>(COUNT);
    for (int i = 0; i < b.size(); ++i) b[i] = TYPE(float(i % 64 + 1));

    auto out = vector<OTYPE>(COUNT, TYPE(0.0f));

    TYPE *d_a;
    TYPE *d_b;
    TYPE *d_o;
    const size_t asize = a.size() * sizeof(TYPE);
    const size_t bsize = b.size() * sizeof(TYPE);
    const size_t osize = out.size() * sizeof(OTYPE);
    auto stream = cudaStream_t(0);

    cudaMalloc( (void**)&d_a, asize ); 
    cudaMalloc( (void**)&d_b, bsize ); 
    cudaMalloc( (void**)&d_o, osize ); 

    cudaMemcpy( d_a, &a[0], asize, cudaMemcpyHostToDevice ); 
    cudaMemcpy( d_b, &b[0], bsize, cudaMemcpyHostToDevice ); 

    //-----------------------------------------
    // Add
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto status = srtAddFlat(DATYPE, d_a, d_b, d_o, COUNT, stream);
        assert(status == cudaSuccess);
    }
    cudaMemcpy( &out[0], d_o, osize, cudaMemcpyDeviceToHost ); 
    auto end = chrono::steady_clock::now();
    auto e = chrono::duration_cast<chrono::microseconds>(end - start).count();
    auto elapsed = float(e) / float(1e6f);
    printf("flat elapsed: %f\n\n", elapsed);

    // for (int i = 0; i < min(COUNT, 10); ++i) cout << "[" << float(out[i].x) << "," << float(out[i].y) << "]" << ", ";
    for (int i = 0; i < min(COUNT, 10); ++i) cout << float(out[i]) << ", ";
    cout << endl;
    cout << endl;

    //-----------------------------------------
    // Add
    start = chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        const size_t shape = COUNT;
        const size_t stride = 1; 
        auto aDesc = srtTensorDescriptor();
        aDesc.type = DATYPE;
        aDesc.rank = 1;
        aDesc.order = CUBLASLT_ORDER_ROW;
        aDesc.count = COUNT;
        aDesc.spanCount = COUNT;
        aDesc.shape = &shape;
        aDesc.strides = &stride;
        aDesc.logicalStrides = &stride;

        auto status = srtAdd(d_a, &aDesc, d_b, &aDesc, d_o, &aDesc, stream);
    }
    cudaMemcpy( &out[0], d_o, osize, cudaMemcpyDeviceToHost ); 
    end = chrono::steady_clock::now();
    e = chrono::duration_cast<chrono::microseconds>(end - start).count();
    elapsed = float(e) / float(1e6f);
    printf("strided elapsed: %f\n\n", elapsed);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);

    // for (int i = 0; i < min(COUNT, 10); ++i) cout << "[" << float(out[i].x) << "," << float(out[i].y) << "]" << ", ";
    for (int i = 0; i < min(COUNT, 10); ++i) cout << float(out[i]) << ", ";
    cout << endl;
    return EXIT_SUCCESS;
}
