/****************************************************************************
 *
 * opencl-dot.c - Dot product
 *
 * Copyright (C) 2017--2021, 2024 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla/)
% Last updated: 2024-11-14

## Familiarize with the environment

The server has three identical GPUs (NVidia GeForce GTX 1070). The
first one is used by default, although it is possible to select
another card using the environment variable `SCL_DEFAULT_DEVICE`.

For example

        SCL_DEFAULT_DEVICE=2 ./opencl-stencil1d

runs `opencl-stencil1d` on device 2; the `sclInitFromFile()` or
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

1. The GPU executes a single 1D workgroup; use the maximum number of
   work-items per workgroup supported by the hardware, which is
   _SCL_DEFAULT_WG_SIZE_.

2. The 2workgroup defines a float array `tmp[]` of length
   _SCL_DEFAULT_WG_SIZE_ in local memory.

3. Work-item $t$ ($t = 0, \ldots, \mathit{BLKDIM}-1$) computes $(x[t]
   \times y[t] + x[t + \mathit{BLKDIM}] \times y[t + \mathit{BLKDIM}]
   + x[t + 2 \times \mathit{BLKDIM}] \times y[t + 2 \times
   \mathit{BLKDIM}] + \ldots)$ and stores the result in `tmp[t]` (see
   Figure 1).

4. When all work-items have completed the previous step (hint: use
   `barrier(CLK_LOCAL_MEM_FENCE)`), work-item 0 performs the
   sum-reduction of `tmp[]` and computes the final result that can be
   transferred back to the host.

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
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h)

***/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "simpleCL.h"

void vec_init( float *x, float *y, int n )
{
    const float tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const float ty[] = {1.0, 2.0, 4.0};
    const size_t arrlen = sizeof(tx)/sizeof(tx[0]);

    for (int i=0; i<n; i++) {
        x[i] = tx[i % arrlen];
        y[i] = ty[i % arrlen];
    }
}

int main( int argc, char* argv[] )
{
    const float TOL = 1e-5;
    float *x, *y, result;               /* host copies of x, y, result */
    cl_mem d_x, d_y, d_result;          /* device copies of x, y, result */
    int n = 1024*1024;
    const int max_len = 64 * n;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }

    const size_t size = n * sizeof(*x);
    sclInitFromFile("opencl-dot.cl");
    sclKernel dot_kernel = sclCreateKernel("dot_kernel");

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(size);
    y = (float*)malloc(size);
    vec_init(x, y, n);

    /* Allocate space for device copies of x, y, result */
    d_x = sclMallocCopy(size, x, CL_MEM_READ_ONLY);
    d_y = sclMallocCopy(size, y, CL_MEM_READ_ONLY);
    d_result = sclMalloc(sizeof(result), CL_MEM_WRITE_ONLY);

    /* Launch dot() kernel on the device */
    printf("Computing the dot product of %d elements... ", n);
    sclSetArgsEnqueueKernel(dot_kernel,
                            DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :b :d :b",
                            d_x, d_y, n, d_result);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(&result, d_result, sizeof(result));

    printf("result=%f\n", result);
    const float expected = ((float)n)/64;

    /* Check result */
    if ( fabsf(result - expected) < TOL ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x); free(y);
    sclFree(d_x); sclFree(d_y); sclFree(d_result);
    sclFinalize();
    return EXIT_SUCCESS;
}
