/****************************************************************************
 *
 * cuda-dot.cu - Dot product
 *
 * Copyright (C) 2017--2024 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
% HPC - Dot product
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-11-14

## Familiarize with the environment

The server has three identical GPUs (NVidia GeForce GTX 1070). The
first one is used by default, although it is possible to select
another GPU either programmatically (`cudaSetDevice(0)` uses the
first GPU, `cudaSetDevice(1)` uses the second one, and so on), or
using the environment variable `CUDA_VISIBLE_DEVICES`.

For example

        CUDA_VISIBLE_DEVICES=0 ./cuda-stencil1d

runs `cuda-stencil1d` on the first GPU (default), while

        CUDA_VISIBLE_DEVICES=1 ./cuda-stencil1d

runs the program on the second GPU.

Run `deviceQuery` from the command line to display the hardware
features of the GPUs.

## Scalar product

The program [cuda-dot.cu](cuda-dot.cu) computes the dot product of two
arrays `x[]` and `y[]` of length $n$. Modify the program to use the
GPU, by defining a suitable kernel and modifying the `dot()` function
to use it. The dot product $s$ of two arrays `x[]` and `y[]` is defined as

$$
s = \sum_{i=0}^{n-1} x[i] \times y[i]
$$

In this exercise we implement a simple (although not efficient)
approach where we use a _single_ block of _BLKDIM_ threads.  The
algorithm works as follows:

1. The GPU executes a single 1D thread block containing _BLKDIM_
   threads; use the maximum number of threads per block supported by
   the hardware, which is _BLKDIM = 1024_.

2. The block defines a float array `tmp[]` of length _BLKDIM_ in
   shared memory.

3. Thread $t$ ($t = 0, \ldots, \mathit{BLKDIM}-1$) computes $(x[t]
   \times y[t] + x[t + \mathit{BLKDIM}] \times y[t + \mathit{BLKDIM}]
   + x[t + 2 \times \mathit{BLKDIM}] \times y[t + 2 \times
   \mathit{BLKDIM}] + \ldots)$ and stores the result in `tmp[t]` (see
   Figure 1).

4. When all threads have completed the previous step (hint: use
   `__syncthreads()`), thread 0 performs the sum-reduction of `tmp[]`
   and computes the final result that can be transferred back to the
   host.

![Figure 1](cuda-dot.svg)

Your program must work correctly for any value of $n$, even if it is
not a multiple of _BLKDIM_.

A better way to compute a reduction will be shown in future lectures.

To compile:

        nvcc cuda-dot.cu -o cuda-dot -lm

To execute:

        ./cuda-dot [len]

Example:

        ./cuda-dot

## Files

- [cuda-dot.cu](cuda-dot.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifndef SERIAL
#define BLKDIM 1024

__global__ void dot_kernel( const float *x, const float *y, int n, float *result )
{
    __shared__ float tmp[BLKDIM];
    const int tid = threadIdx.x;
    float s = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        s += x[i] * y[i];
    }
    tmp[tid] = s;
    __syncthreads();
    if (0 == tid) {
        *result = 0;
        for (int i=0; i < blockDim.x; i++) {
            *result += tmp[i];
        }
    }
}
#endif

float dot( const float *x, const float *y, int n )
{
#ifdef SERIAL
    /* [TODO] modify this function so that (part of) the computation
       is executed on the GPU. You may want to follow the steps
       below. */

    /* Define a `float` variabile `result` in host memory */

    /* Allocate space for device copies of `x`, `y` and `result` */

    /* Copy `x`, `y` from host to device */

    /* Launch a suitable kernel on the GPU */

    /* Copy the value of `result` back to host memory */

    /* Perform the final reduction on the CPU */

    /* Free device memory */

    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
#else
    float result;
    float *d_x, *d_y, *d_result; /* device copies of x, y, result */
    const size_t SIZE_XY = n*sizeof(*x);
    const size_t SIZE_RESULT = sizeof(result);

    /* Allocate space for device copies of x, y */
    cudaSafeCall( cudaMalloc((void **)&d_x, SIZE_XY) );
    cudaSafeCall( cudaMalloc((void **)&d_y, SIZE_XY) );
    cudaSafeCall( cudaMalloc((void **)&d_result, SIZE_RESULT) );

    /* Copy inputs to device memory */
    cudaSafeCall( cudaMemcpy(d_x, x, SIZE_XY, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_y, y, SIZE_XY, cudaMemcpyHostToDevice) );

    /* Launch dot_kernel() on GPU */
    dot_kernel<<<1, BLKDIM>>>(d_x, d_y, n, d_result);
    cudaCheckError();

    /* Copy the result back to host memory */
    cudaSafeCall( cudaMemcpy(&result, d_result, SIZE_RESULT, cudaMemcpyDeviceToHost) );

    /* Free device memory */
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return result;
#endif
}

/**
 * Initialize `x` and `y` of length `n`; return the expected dot
 * product of `x` and `y`. To avoid numerical issues, the vectors are
 * initialized with integer values, so that the result can be computed
 * exactly (save for possible overflows, which should not happen
 * unless the vectors are very long).
 */
float vec_init( float *x, float *y, int n )
{
    const float tx[] = {1, 2, -5};
    const float ty[] = {1, 2, 1};

    const size_t LEN = sizeof(tx)/sizeof(tx[0]);
    const float expected[] = {0, 1, 5};

    for (int i=0; i<n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }

    return expected[n % LEN];
}


int main( int argc, char* argv[] )
{
    float *x, *y, result;
    int n = 1024*1024;
    const int MAX_N = 128 * n;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( (n < 0) || (n > MAX_N) ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(SIZE);
    assert(x != NULL);
    y = (float*)malloc(SIZE);
    assert(y != NULL);
    const float expected = vec_init(x, y, n);

    printf("Computing the dot product of %d elements...\n", n);
    result = dot(x, y, n);
    printf("got=%f, expected=%f\n", result, expected);

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED\n");
    }

    /* Cleanup */
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
