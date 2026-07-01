/****************************************************************************
 *
 * opencl-rank.c - Rank elements of an array.
 *
 * Copyright (C) 2026 Moreno Marzolla
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
% Rank elements of an array
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-07-01

Given an array $v$ of length $n$, the _rank_ $r[i]$ of $v[i]$ if the
number of elements that are lower than $v[i]$; this definition assumes
that $v$ does not contain duplicate values, but this program handles
the general case where this might not be true. $r[i]$ is the position
that $v[i]$ would occupy if the array $v$ were sorted; therefore, the
array of ranks defines a sorting permutation of $v$.

From the discussion above, it is possible to compute the ranks by
simply sorting $v$ and keeping track of the sorting permutation. This
can be accomplished in $\Theta(n \log n)$ serial time using an
efficient general-purpose sorting algorithm.

The goal of this exercise is to write a CUDA version of the trivial
ranking algorithm that works by comparing each element $v[i]$ with all
other elements, and count how many of them are lower than $v[i]$.

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-rank.c simpleCL.c -o opencl-rank -lOpenCL

To execute:

        ./opencl-rank

## Files

- [opencl-rank.c](opencl-rank.c)

***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "simpleCL.h"
#include "hpc.h"

#ifndef SERIAL
sclKernel rank_kernel;
#endif

void rank(const int *v, int *r, int n)
{
#ifdef SERIAL
    for (int i=0; i<n; i++) {
        r[i] = 0;
        for (int j=0; j<n; j++) {
            if ((v[i] > v[j]) || (v[i] == v[j] && i < j))
                r[i]++;
        }
    }
#else
    cl_mem d_v, d_r;
    const size_t SIZE = n * sizeof(int);
    d_v = sclMallocCopy(SIZE, (int*)v, CL_MEM_READ_ONLY);
    d_r = sclMalloc(SIZE, CL_MEM_READ_WRITE);
    sclSetArgsEnqueueKernel(rank_kernel,
                            DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE)),
                            DIM1(SCL_DEFAULT_WG_SIZE),
                            ":b :b :d",
                            d_v, d_r, n);
    sclMemcpyDeviceToHost(r, d_r, SIZE);
    sclFree(d_v);
    sclFree(d_r);
#endif
}

int main( int argc, char *argv[])
{
    int *v = NULL, *r = NULL;
#ifndef SERIAL
    sclInitFromFile("opencl-rank.cl");
    rank_kernel = sclCreateKernel("rank_kernel");
#endif
    int n = 100*SCL_DEFAULT_WG_SIZE;

    if (argc > 1)
        n = atoi(argv[1]);

    printf("Ranking %d elements\n", n);

    /* Allocate host arrays. */
    v = (int*)malloc( n * sizeof(*v)); assert(v != NULL);
    r = (int*)malloc( n * sizeof(*r)); assert(r != NULL);

    for (int i=0; i<n; i++) {
        v[i] = i;
    }

    const double tstart = hpc_gettime();
    rank(v, r, n);
    const double elapsed = hpc_gettime() - tstart;

    printf("Execution time %.3f\n", elapsed);

    for (int i=0; i<n; i++) {
        if (r[i] != i) {
            printf("FATAL: rank[%d] == %d, expected %d\n", i, r[i], i);
            return EXIT_FAILURE;
        }
    }
    printf("Check OK\n");

    free(v);
    free(r);

#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
