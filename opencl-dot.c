/****************************************************************************
 *
 * opencl-dot.c - Dot product
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Last updated: 2023-06-07

## Familiarize with the environment

The server has three identical GPUs (NVidia GeForce GTX 1070). The
first one is used by default, although it is possible to select
another card using the environment variable `SCL_DEFAULT_DEVICE`.

For example

        SCL_DEFAULT_DEVICE=2 ./opencl-stencil1d

runs `cuda-stencil1d` on device 2; the `sclInitFromFile()` or
`sclInitFromString()` functions print the list of available devices,
as does the `clinfo` command-line tool.

## Scalar product

The program [opencl-dot.c](opencl-dot.c) computes the dot product of two
arrays `x[]` and `y[]` of length $n$. Modify the program to use the
GPU, by transforming the `dot()` function into a kernel.  The dot
product $s$ of two arrays `x[]` and `y[]` is defined as

$$
s = \sum_{i=0}^{n-1} x[i] \times y[i]
$$

Some modifications of the `dot()` function are required to use the
GPU. In this exercise we implement a simple (although not efficient)
approach where we use a _single_ workgroup of _SCL_DEFAULT_WG_SIZE_
work-items.  The algorithm works as follows:

1. The CPU allocates a `tmp[]` array of $B :=
   \mathit{SCL\_DEFAULT\_WG\_SIZE}$ elements on the GPU, in addition
   to a copy of `x[]` and `y[]`.

2. The CPU executes _B_ work-items; use the maximum number of
   work-items per workgroup supported by the hardware.

3. Work-item $t$ ($t = 0, \ldots, B-1$) computes the value of the
   expression $(x[t] \times y[t] + x[t + B] \times y[t + B] + x[t +
   2B] \times y[t + 2B] + \ldots)$ and stores the result in `tmp[t]`
   (see Figure 1).

4. When the kernel terminates, the CPU transfers `tmp[]` back to host
   memory and performs a sum-reduction to compute the final result.

![Figure 1](opencl-dot.svg)

Your program must work correctly for any value of $n$, even if it is
not a multiple of _B_.

A better way to compute a reduction will be shown in future lectures.

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-dot.c simpleCL.c -o opencl-dot -lOpenCL

To execute:

        ./opencl-dot [len]

Example:

        ./opencl-dot

## Files

- [opencl-dot.c](opencl-dot.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "simpleCL.h"

#ifndef SERIAL
const char *program =
"__kernel void dot_kernel( __global const float *x,\n"
"                          __global const float *y,\n"
"                          int n,\n"
"                          __global float *tmp )\n"
"{\n"
"    const int tid = get_local_id(0);\n"
"    const int local_size = get_local_size(0);\n"
"    int i;\n"
"    float s = 0.0;\n"
"    for (i = tid; i < n; i += local_size) {\n"
"        s += x[i] * y[i];\n"
"    }\n"
"    tmp[tid] = s;\n"
"}\n";

sclKernel dot_kernel;
#endif

float dot( float *x, float *y, int n )
{
#ifdef SERIAL
    /* [TODO] modify this function so that (part of) the dot product
       computation is executed on the GPU. */
    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
#else
    float tmp[SCL_DEFAULT_WG_SIZE];
    cl_mem d_x, d_y, d_tmp; /* device copies of x, y, tmp */
    const size_t SIZE_TMP = sizeof(tmp);
    const size_t SIZE_XY = n*sizeof(*x);

    /* Allocate space for device copies of x, y */
    d_x = sclMallocCopy(SIZE_XY, x, CL_MEM_READ_ONLY);
    d_y = sclMallocCopy(SIZE_XY, y, CL_MEM_READ_ONLY);
    d_tmp = sclMalloc(SIZE_TMP, CL_MEM_WRITE_ONLY);

    /* Launch dot_kernel() on the GPU */
    sclSetArgsLaunchKernel(dot_kernel,
                           DIM1(SCL_DEFAULT_WG_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :b :d :b",
                           d_x, d_y, n, d_tmp);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(tmp, d_tmp, SIZE_TMP);

    /* Perform the last reduction on the CPU */
    float result = 0.0;
    for (int i=0; i<SCL_DEFAULT_WG_SIZE; i++) {
        result += tmp[i];
    }

    /* Cleanup */
    sclFree(d_x);
    sclFree(d_y);
    sclFree(d_tmp);

    return result;
#endif
}

void vec_init( float *x, float *y, int n )
{
    int i;
    const float tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const float ty[] = {1.0, 2.0, 4.0};
    const size_t LEN = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }
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

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

#ifndef SERIAL
    sclInitFromString(program);
    dot_kernel = sclCreateKernel("dot_kernel");
#endif
    const size_t SIZE = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(SIZE);
    assert(x != NULL);
    y = (float*)malloc(SIZE);
    assert(y != NULL);
    vec_init(x, y, n);

    printf("Computing the dot product of %d elements... ", n);
    result = dot(x, y, n);
    printf("result=%f\n", result);

    const float expected = ((float)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x);
    free(y);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
