/****************************************************************************
 *
 * opencl-merge-sort.c - Bottom-up Merge Sort with OpenCL
 *
 * Copyright (C) 2017--2026 Moreno Marzolla
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
% Bottom-up Merge Sort with OpenCL
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-05-27

The goal of this exercise is to implement a bottom-up, iterative
version of the _Merge Sort_ algorithm. Starting with an unsorted array
`v[]` of length $n$, The idea is to merge adjacent subvectors of `v[]`
of increasing lengths `len`, for `len = 1, 2, 4, 8, ...`.

To compile:

        gcc -std=c99 -Wall -Wpedantic opencl-merge-sort.c simpleCL.c -o opencl-merge-sort -lOpenCL

To sort an array of length $n$:

        ./opencl-merge-sort [n]

Example:

        ./opencl-merge-sort 500000

## Files

- [opencl-merge-sort.c](opecnl-merge-sort.c)

***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "simpleCL.h"
#include "hpc.h"

void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void print_array(const int *v, int n)
{
    for (int i=0; i<n; i++) {
        printf("%d ",v[i]);
    }
    printf("\n");
}

#ifdef SERIAL
int min(int a, int b)
{
    return (a < b ? a : b);
}

/**
 * Merge two adjacent sub-arrays `src[low..mid]` and
 * `src[mid+1..high]`, put the result in `dst[low..high]`.
 */
void merge(const int* src, int low, int mid, int high, int* dst)
{
    int i=low, j=mid+1, k=low;
    while (i<=mid && j<=high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle leftovers */
    while (i<=mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j<=high) {
        dst[k] = src[j++];
        k++;
    }
}
#else
sclKernel merge_kernel;
#endif

/**
 *
 */
void bottom_up_mergesort(int* v, int n)
{
#ifdef SERIAL
    /* double-buffering. */
    int *tmp = (int*)malloc(n * sizeof(*v)); assert(tmp != NULL);
    int *buf[2] = {v, tmp};
    int cur = 0, next = 1-cur; /* current array to be sorted. */

    for (int len=1; len < n; len *= 2) {
        /* merge adjacent sub-arrays of length `len`. */
        for (int i=0; i<n; i += 2*len) {
            const int m = min(n, i+len);
            const int j = min(n, m+len);
            merge(buf[cur], i, m-1, j-1, buf[next]);
        }
        cur = next;
        next = 1-cur;
    }
    if (buf[cur] != v)
        memcpy(v, buf[cur], n*sizeof(*v));
    free(tmp);
#else
    /* double-buffering. */
    cl_mem buf[2];
    const size_t SIZE = n*sizeof(*v);
    buf[0] = sclMallocCopy(SIZE, v, CL_MEM_READ_WRITE);
    buf[1] = sclMalloc(SIZE, CL_MEM_READ_WRITE);
    int cur = 0, next = 1-cur; /* current array to be sorted. */

    for (int len=1; len < n; len *= 2) {
        /* merge adjacent sub-arrays of length `len`. */
        const int nthreads = (n + 2*len-1)/(2*len);
        sclDim grid, block;
        sclWGSetup1D(nthreads, &grid, &block);
        sclSetArgsEnqueueKernel(merge_kernel,
                                grid, block,
                                ":b :d :d :b",
                                buf[cur],
                                len,
                                n,
                                buf[next],
                                merge_kernel);
        cur = next;
        next = 1-cur;
    }
    sclMemcpyDeviceToHost(v, buf[cur], SIZE);
    sclFree(buf[0]);
    sclFree(buf[1]);
#endif
}

/* Returns a random integer in the range [a..b], inclusive */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/**
 * Fills a[] with a random permutation of the intergers 0..n-1; the
 * caller is responsible for allocating a
 */
void fill(int* a, int n)
{
    for (int i=0; i<n; i++) {
        a[i] = i;
    }
    for (int i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        swap(a+i, a+j);
    }
}

/* Return 1 iff a[] contains the values 0, 1, ... n-1, in that order */
int is_correct(const int* a, int n)
{
    for (int i=0; i<n; i++) {
        if ( a[i] != i ) {
            fprintf(stderr, "Expected a[%d]=%d, got %d\n", i, i, a[i]);
            return 0;
        }
    }
    return 1;
}

int main( int argc, char* argv[] )
{
    int n = 10000000;

#ifndef SERIAL
    sclInitFromFile("opencl-merge-sort.cl");
    merge_kernel = sclCreateKernel("merge_kernel");
#endif
    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if (n > 1000000000) {
        fprintf(stderr, "FATAL: array too large\n");
        return EXIT_FAILURE;
    }

    int *a = (int*)malloc(n*sizeof(a[0]));
    assert(a != NULL);

    printf("Initializing array...\n");
    fill(a, n);
    printf("Sorting %d elements...", n); fflush(stdout);
    const double tstart = hpc_gettime();
    bottom_up_mergesort(a, n);
    const double elapsed = hpc_gettime() - tstart;
    printf("done\n");
    const int ok = is_correct(a, n);
    printf("Check %s\n", (ok ? "OK" : "failed"));
    printf("Execution time %.3f\n", elapsed);

    free(a);

#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
