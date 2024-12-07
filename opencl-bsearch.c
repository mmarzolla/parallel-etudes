/****************************************************************************
 *
 * opencl-bsearch.c - Generalized binary search
 *
 * Copyright (C) 2022--2024 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Ceneralized binary search
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-11-15

Implementation of CREW Search, p. 115 of Selim G. Akl, _The Design and
Analysis of Parallel Algorithms_, Prentice-Hall International Editions, 1989,
ISBN 0-13-200073-3

Compile with:

        cc opencl-bsearch.c simpleCL.c -o opencl-bsearch -lOpenCL

Run with:

        ./opencl-bsearch [len [key]]

Example:

        ./opencl-bsearch

## Files

- [opencl-bsearch.c](opencl-bsearch.c)

***/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

void vec_init( int *x, int n )
{
    int i;

    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

/* Returns the position of `key` in the sorted array `x[]` of length
   `n`. Returns -1 if `key` not found. */
int seq_bsearch(const int *x, int n, int key)
{
    const int P = 4096;
    int cmp[P];
    size_t m[P];
    size_t i;
    int start = 0, end = n-1;
    while (end-start > P) {
        printf("start=%d end=%d\n", (int)start, (int)end);
        for (i=0; i<P; i++) {
            m[i] = start + ((end-start)*i + P)/(P+1);
            printf("m[%d]=%d ", (int)i, (int)m[i]);
        }
        printf("\n");
        for (i=0; i<P; i++) {
            cmp[i] = (x[m[i]] < key ? 1 : -1);
            printf("cmp[%d]=%d ", (int)m[i], cmp[i]);
        }
        printf("\n");
        /* assertion:

           cmp[i] == 1 -> if key is present, it is in position > m[i]
           cmp[i] == -1 -> if key is present, it is in position <= m[i] */
        i=0;
        while (i<P && cmp[i]>0)
            i++;

        /* assertion:

           i is the smallest index s.t. cmp[i] < 0 */

        if (i==0)
            end = m[0];
        else if (i==P)
            start = m[P-1]+1;
        else {
            start = m[i-1]+1;
            end = m[i];
        }
    }
    for (i=start; i<=end; i++) {
        if (x[i] == key)
            return i;
    }
    return -1;
}

int main( int argc, char* argv[] )
{
    int n = 1024*1024;
    int *x, result, key = n/2;          /* host copies of x, y, result */
#ifndef SERIAL
    cl_mem d_x, d_result;               /* device copies of x, y, result */
#endif
    const int max_len = n * 64;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [len [key]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }

    if ( argc > 2 ) {
        key = atoi(argv[2]);
    }
    const size_t size = n * sizeof(*x);

    /* Allocate space for host copies of x */
    x = (int*)malloc(size); assert(x != NULL);
    vec_init(x, n);

    printf("Searching for %d on %d elements... ", key, n);
#ifdef SERIAL
    result = seq_bsearch(x, n, key);
#else
    sclInitFromFile("opencl-bsearch.cl");
    sclKernel bsearch_kernel = sclCreateKernel("bsearch_kernel");

    /* Allocate space for device copies of x, result */
    d_x = sclMallocCopy(size, x, CL_MEM_READ_ONLY);
    d_result = sclMalloc(sizeof(result), CL_MEM_WRITE_ONLY);

    /* Launch bsearch() kernel on the device */
    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE);
    const sclDim GRID = DIM1(SCL_DEFAULT_WG_SIZE);
    sclSetArgsEnqueueKernel(bsearch_kernel,
                            GRID, BLOCK,
                            ":b :d :d :b",
                            d_x, n, key, d_result);
    /* Copy result back to host */
    sclMemcpyDeviceToHost(&result, d_result, sizeof(result));
#endif

    printf("result=%d\n", result);
    const int expected = (key < 0 || key >= n ? -1 : key);

    /* Check result */
    if ( result == expected ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %d, expected %d\n", result, expected);
    }

    /* Cleanup */
    free(x);
#ifndef SERIAL
    sclFree(d_x); sclFree(d_result);
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
