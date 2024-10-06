/****************************************************************************
 *
 * opencl-reverse.c - Array reversal with OpenCL
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Array reversal with OpenCL
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

Write a program that reverses an array `v[]` of length $n$, i.e.,
exchanges `v[0]` and `v[n-1]`, `v[1]` and `v[n-2]` and so
on. You should write two versions of the program:

1. the first version reverses an input array `in[]` into a different
   output array `out[]`, so that the input is not modified. You can
   assume that `in[]` and `out[]` are mapped to different,
   non-overlapping memory blocks.

2. The second version reverses an array `in[]` "in place" using $O(1)$
   additional storage.

The file [opencl-reverse.c](opencl-reverse.c) provides a CPU-based
implementation of `reverse()` and `inplace_reverse()`.  Modify the
functions to use of the GPU.

**Hint:** `reverse()` can be easily transformed into a kernel executed
by $n$ work-items (one for each array element). Each work-item copies
one element from `in[]` to `out[]`. Use one-dimensional workgroups,
since that makes easy to map work-itemss to array elements.
`inplace_reverse()` can be transformed into a kernel as well, but in
this case only $\lfloor n/2 \rfloor$ work-items are required (note the
rounding): each work-item swaps an element from the first half of
`in[]` with the appropriate element from the second half. Make sure
that the program works also when the input length $n$ is odd.

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-reverse.c simpleCL.c -o opencl-reverse -lOpenCL

To execute:

        ./opencl-reverse [n]

Example:

        ./opencl-reverse

## Files

- [opencl-reverse.c](opencl-reverse.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

/* Reverse in[] into out[].

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in` and `out`;
   - copies `in` and `out` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
 */
void reverse( int *in, int *out, int n )
{
#ifdef SERIAL
    int i;
    for (i=0; i<n; i++) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
#else
    cl_mem d_in, d_out; /* device copy of in and out */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] and out[] */
    d_in = sclMallocCopy(SIZE, in, CL_MEM_READ_ONLY);
    d_out = sclMalloc(SIZE, CL_MEM_WRITE_ONLY);

    /* Launch the reverse() kernel on the GPU */
    sclSetArgsLaunchKernel(sclCreateKernel("reverse_kernel"),
                           DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :b :d",
                           d_in, d_out, n);

    /* Copy the result back to host memory */
    sclMemcpyDeviceToHost(out, d_out, SIZE);

    /* Free memory on the device */
    sclFree(d_in);
    sclFree(d_out);
#endif
}

/* In-place reversal of in[] into itself.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
*/
void inplace_reverse( int *in, int n )
{
#ifdef SERIAL
    int i = 0, j = n-1;
    while (i < j) {
        const int tmp = in[j];
        in[j] = in[i];
        in[i] = tmp;
        j--;
        i++;
    }
#else
    cl_mem d_in; /* device copy of in */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] */
    d_in = sclMallocCopy(SIZE, in, CL_MEM_READ_WRITE);

    /* Launch the reverse() kernel on the GPU */
    sclSetArgsLaunchKernel(sclCreateKernel("inplace_reverse_kernel"),
                           DIM1(sclRoundUp(n/2, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :d",
                           d_in, n);

    /* Copy the result back to host memory */
    sclMemcpyDeviceToHost(in, d_in, SIZE);

    /* Free memory on the device */
    sclFree(d_in);
#endif
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
#ifndef SERIAL
    sclInitFromFile("opencl-reverse.cl");
#endif
    int *in, *out;
    int n = 1024*1024;
    const int MAX_N = 512*1024*1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int*)malloc(SIZE); assert(in != NULL);
    out = (int*)malloc(SIZE); assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
