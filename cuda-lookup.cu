/****************************************************************************
 *
 * cuda-lookup.cu - Parallel linear search
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
% HPC - Parallel linear search
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-09-06

Write an CUDA program that finds the positions of all occurrences of a
given `key` in an unsorted integer array `v[]`. For example, if `v[] =
{1, 3, -2, 3, 4, 3, 3, 5, -10}` and `key = 3`, the program must
build the result array

        {1, 3, 5, 6}
To compile:

        nvcc cuda-lookup.cu -o cuda-lookup -lm

To execute:

        ./cuda-lookup [N]

Example:

        ./cuda-lookup [N]

## Files

- [cuda-lookup.cu](cuda-lookup.cu)

***/

#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <stdio.h>
#include <assert.h>
#include "hpc.h"

#define BLKDIM 1024

#ifdef SERIAL
#else
__global__ void count_kernel (int *v, int n, int *nf, int KEY) {
    __shared__ int temp[BLKDIM];
    const int tid = threadIdx.x;
    const int i = tid + blockIdx.x * blockDim.x;
    temp[tid] = 0;                                                                                                                                                                                                                                                                         

    if (i < n && v[i] == KEY) {
        temp[tid]++;
    } 
    __syncthreads(); 

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            temp[tid] += temp[tid + offset];
        }
    __syncthreads(); 
    }
    
    if (tid == 0) {
        atomicAdd(nf, temp[0]);
    }   
}

__global__ void find_indexes_kernel (int *v, int *result, int n, int nf, int KEY, int *r) {
    const int tid = threadIdx.x;
    const int i = tid + blockIdx.x * blockDim.x;

    if (v[i] == KEY && *r < nf && i < n) {
        int idx = atomicAdd(r, 1);
        result[idx] = i;
    }

}
#endif

void fill (int *v, int n) {
    int i;
    for (i = 0; i < n; i++) {
        v[i] = (rand() % 100);
    }
}

int main (int argc, char *argv[]) {

    int n = 1000;       /* Lenght of the input array */
    int *v = NULL;      /* Input array */
    int *result = NULL; /* Array which contains the indexes of the occurrences */
    int nf = 0;         /* Total occurrences */
    const int KEY = 42; /* The value to search for */

    if (BLKDIM & (BLKDIM-1) != 0) {
        fprintf(stderr, "BLKDIM must be a power of two\n");
        return EXIT_FAILURE;
    } 

    if (argc > 1){
        n = atoi(argv[1]);

        v = (int*)malloc(n * sizeof(*v)); 
        assert(v != NULL);
        fill(v, n);
    }

    const double tstart = hpc_gettime();
#ifdef SERIAL
    /* Counts the number of occurrences of `KEY` in `v[]` */
    
    for (int i = 0; i < n; i++) {
        if (v[i] == KEY){
            nf++;
        }
    }

    /* Allocates the result array */
    result = (int*) malloc(nf * sizeof(*result)); 
    assert(result != NULL);

    /* Fills the result array  */
    int r = 0;
    for (int i = 0; i < n; i++) {
        if (v[i] == KEY) {
            result[r] = i;
            r++;
        }
    }
#else

    int *d_v, *d_result, *d_nf, *d_r = 0;
    const size_t size = n * sizeof(*v);
    const int n_of_blocks = (n + BLKDIM - 1) / BLKDIM;

    cudaMalloc((void **)&d_v, size);
    cudaMalloc((void **)&d_nf, sizeof(*d_nf));
    cudaMalloc((void **)&d_r, sizeof(*d_r));

    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nf, &nf, sizeof(nf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, &nf, sizeof(nf), cudaMemcpyHostToDevice);

    count_kernel <<<n_of_blocks, BLKDIM>>> (d_v, n, d_nf, KEY);

    cudaMemcpy(&nf, d_nf, sizeof(nf), cudaMemcpyDeviceToHost);
    cudaMalloc((void **)&d_result, nf * sizeof(result));

    find_indexes_kernel <<<n_of_blocks, BLKDIM>>> (d_v, d_result, n, nf, KEY, d_r);
    result = (int*) malloc(nf * sizeof(*result)); 
    assert(result != NULL);
    cudaMemcpy(result, d_result, nf * sizeof(*result), cudaMemcpyDeviceToHost);
    
    cudaFree(d_v); 
    cudaFree(d_nf);
    cudaFree(d_r);
    cudaFree(d_result);
    cudaDeviceSynchronize();
#endif
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

    printf("There are %d occurrences of %d\n", nf, KEY);
    printf("Positions: ");
    for (int i = 0; i < nf; i++) {
        printf("%d ", result[i]);
        if (v[result[i]] != KEY) {
            fprintf(stderr, "\nFATAL: v[%d]=%d, expected %d\n", result[i], v[result[i]], KEY);
        }
    }
    printf("\n");
    free(v);
    free(result);

    return EXIT_SUCCESS;
}