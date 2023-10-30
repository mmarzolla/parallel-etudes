/****************************************************************************
 *
 * cuda-sum.cu - Sum-reduction of an array
 *
 * Copyright (C) 2023 by Alice Girolomini <alice.girolomini(at)studio.unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Sum-reduction of an array
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-10-30

The file [cuda-sum.cu](cuda-sum.cu) contains a serial implementation of a
CUDA program that computes the sum of an array of length $N$; indeed,
the program performsa a sum-reduction of the array 

To compile:

        nvcc cuda-sum.cu -o cuda-sum -lm

To execute:

        ./cuda-sum [N]

Example:

        ./cuda-sum 2048

## Files

- [cuda-sum.cu](cuda-sum.cu)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hpc.h"

#define BLKDIM 1024

#ifdef SERIAL
/** 
 * Computes the sum of all elements of array `v` of length `n` 
*/
float sum (float *v, int n) {
    float sum = 0;
    int i;

    for (i=0; i<n; i++) {
        sum += v[i];
    }

    return sum;
}

/**
 * Fills the array `v` of length `n`; returns the sum of the
 * content of `v`
*/
float fill (float *v, int n) {
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);
    int i;

    
    for (i = 0; i < n; i++) {
        v[i] = vals[i % NVALS];
    }

    switch(i % NVALS) {
    case 1: return 1; break;
    case 3: return 2; break;
    default: return 0;
    }
}

#else
/** 
 * All threads within the block cooperate to compute the local sum,
 * then thread 0 of each block performs an atomic addition
*/
__global__ void sum (float *v, int n, float *s) {
    __shared__ float temp[BLKDIM];
    const int tid = threadIdx.x;
    const int i = tid + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    if (i < n) {
        temp[tid] = v[i];
    } else {
        temp[tid] = 0;
    }
    __syncthreads(); 

    while (bsize > 0) {
        if (tid < bsize) {
            temp[tid] += temp[tid + bsize];
        }
        bsize /=  2; 
        __syncthreads(); 
    }

    if (tid == 0) {
        atomicAdd(s, temp[0]);
    }
}

/** 
 * Each thread fills its portion of the array `v` of length `n`; 
 * returns the expected result of the reduction 
*/
__global__ void fill (float *v, int n, float *expected) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int block = blockIdx.x;
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);

    if (i < n) {
        v[i] = vals[i % NVALS];
    }
    
    if (i == 0 && block == 0) {
        switch(n % NVALS) {
            case 1: atomicExch(expected, 1); break;
            case 3: atomicExch(expected, 2); break;
            default: atomicExch(expected, 0);
        }
    }
}
#endif

int main (int argc, char *argv[]) {

    int n = 1024 * 10; 
    float s = 0, expected;
    float *v;

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (BLKDIM & (BLKDIM-1) != 0) {
        fprintf(stderr, "BLKDIM must be a power of two\n");
        return EXIT_FAILURE;
    } 

    v = (float*) malloc(n * sizeof(float));
    assert(v != NULL);

    const double tstart = hpc_gettime();
#ifdef SERIAL
    expected = fill(v, n);
    s = sum(v, n);
#else
    float *d_v, *d_s, *d_expected;
    const size_t size = n * sizeof(*v);
    const int n_of_blocks = (n + BLKDIM - 1) / BLKDIM;

    /**
     * Allocates space for device copies of v, s, expected
    */
    cudaMalloc((void **)&d_v, size);
    cudaMalloc((void **)&d_s, sizeof(*d_s));
    cudaMalloc((void **)&d_expected, sizeof(*d_expected));

    cudaMemcpy(d_s, &s, sizeof(s), cudaMemcpyHostToDevice);

    fill <<<n_of_blocks, BLKDIM>>> (d_v, n, d_expected);
    sum <<<n_of_blocks, BLKDIM>>> (d_v, n, d_s);

    /**
     * Copies the result from device memory to host memory
    */
    cudaMemcpy(&expected, d_expected, sizeof(expected), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s, d_s, sizeof(s), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
#endif
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

    printf("Sum=%f, expected=%f\n", s, expected);
    if (s == expected) {
        printf("Test OK\n");
    } else {
        printf("Test FAILED\n");
    }
    
#ifdef SERIAL
    free(v);
#else
    free(v);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_expected);
#endif 
    return EXIT_SUCCESS;
}