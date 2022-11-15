/****************************************************************************
 *
 * cuda-dot.cu - Dot product
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-11-11

## Familiarize with the environment

The server has three identical GPUs (NVidia GeForce GTX 1070). The
first one is used by default, although it is possible to select
another card either programmatically (e.g., `cudaSetDevice(0)` uses
the first GPU, `cudaSetDevice(1)` uses the second one, and so on), or
using the environment variable `CUDA_VISIBLE_DEVICES`.

For example

        CUDA_VISIBLE_DEVICES=0 ./cuda-stencil1d

runs `cuda-stencil1d` on the first GPU (default), while

        CUDA_VISIBLE_DEVICES=1 ./cuda-stencil1d

runs the program on the second one.

Invoke `deviceQuery` from the command line to display the hardware
features of the GPUs.

## Scalar product

The program [cuda-dot.cu](cuda-dot.cu) computes the dot product of two
arrays `x[]` and `y[]` of equal length $n$. Modify the program to use
the GPU, by transforming the `dot()` function into a kernel.  The dot
product $s$ of two arrays `x[]` and `y[]` is defined as

$$
s = \sum_{i=0}^{n-1} x[i] \times y[i]
$$

Some modifications of the `dot()` function are required to use the
GPU. In this exercise we implement a simple (although not very
efficient) approach where we use a single block of _BLKDIM_ threads:

1. The CPU allocates a `tmp[]` array of _BLKDIM_ elements on the GPU,
   in addition to a copy of `x[]` and `y[]`. Use `cudaMalloc()` to
   allocate `tmp[]`.

2. The CPU executes a single 1D thread block consisting of _BLKDIM_
   threads

3. Thread $t$ ($t = 0, \ldots, \mathit{BLKDIM}-1$) computes the value
   of the expression $(x[t] \times y[t] + x[t + \mathit{BLKDIM}]
   \times y[t + \mathit{BLKDIM}] + x[t + 2 \times \mathit{BLKDIM}]
   \times y[t + 2 \times \mathit{BLKDIM}] + \ldots)$ and stores the
   result in `tmp[t]` (see Figure 1).

4. When the kernel terminates, the CPU transfers `tmp[]` into host
   memory and performs a sum-reduction to compute the result.

The computation of the dot product requires a step (step 3) that is
performed by the GPU, and a second step (step 4) that is performed by
the GPU.

![Figure 1](cuda-dot.png)

The program must work correctly for any value of $n$, even if it is
not a multiple of _BLKDIM_.

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

__global__ void dot_kernel( double *x, double *y, int n, double *tmp )
{
    const int tid = threadIdx.x;
    int i;
    double s = 0.0;
    for (i = tid; i < n; i += blockDim.x) {
        s += x[i] * y[i];
    }
    tmp[tid] = s;
}
#endif

double dot( double *x, double *y, int n )
{
#ifdef SERIAL
    /* [TODO] modify this function so that (part of) the dot product
       computation is executed on the GPU. */
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
#else
    double tmp[BLKDIM];
    double *d_x, *d_y, *d_tmp; /* device copies of x, y, tmp */
    const size_t SIZE_TMP = sizeof(tmp);
    const size_t SIZE_XY = n*sizeof(*x);

    /* Allocate space for device copies of x, y */
    cudaSafeCall( cudaMalloc((void **)&d_x, SIZE_XY) );
    cudaSafeCall( cudaMalloc((void **)&d_y, SIZE_XY) );
    cudaSafeCall( cudaMalloc((void **)&d_tmp, SIZE_TMP) );

    /* Copy inputs to device memory */
    cudaSafeCall( cudaMemcpy(d_x, x, SIZE_XY, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_y, y, SIZE_XY, cudaMemcpyHostToDevice) );

    /* Launch dot_kernel() on GPU */
    dot_kernel<<<1, BLKDIM>>>(d_x, d_y, n, d_tmp);
    cudaCheckError();

    /* Copy result back to host memory */
    cudaSafeCall( cudaMemcpy(tmp, d_tmp, SIZE_TMP, cudaMemcpyDeviceToHost) );

    /* Perform the last reduction on the CPU */
    double result = 0.0;
    for (int i=0; i<BLKDIM; i++) {
        result += tmp[i];
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tmp);

    return result;
#endif
}

void vec_init( double *x, double *y, int n )
{
    int i;
    const double tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const double ty[] = {1.0, 2.0, 4.0};
    const size_t LEN = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }
}

int main( int argc, char* argv[] )
{
    double *x, *y, result;
    int n = 1024*1024;
    const int MAX_N = 128 * n;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (double*)malloc(SIZE);
    assert(x != NULL);
    y = (double*)malloc(SIZE);
    assert(y != NULL);
    vec_init(x, y, n);

    printf("Computing the dot product of %d elements... ", n);
    result = dot(x, y, n);
    printf("result=%f\n", result);

    const double expected = ((double)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
