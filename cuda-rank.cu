/****************************************************************************
 *
 * cuda-rank.cu - Rank elements of an array.
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
% Last updated: 2026-06-30

Given an array $v$ of length $n$, the _rank_ $r[i]$ of $v[i]$ if the
number of elements that are lower than $v[i]$ (this definition assumes
that $v$ does not contain duplicate values; however, this program
handles the general case where this is not necessarily true). $r[i]$
is the position that $v[i]$ would occupy if the array $v$ were sorted;
therefore, the array of ranks defines a sorting permutation of $v$.

From the discussion above, it is possible to compute the ranks by
simply sorting $v$ and keeping track of the sorting permutation. This
can be accomplished in $\Theta(n \log n)$ serial time using an
efficient general-purpose sorting algorithm.

The goal of this exercise is to write a distributed-memory version of
the trivial ranking algorithm that works by comparing each element
$v[i]$ with all other elements, and count how many of them are lower
than $v[i]$. The algorithm assumes that $v$ does not contain duplicate
values.

To compile:

        nvcc cuda-rank.cu -o cuda-rank

To execute:

        ./cuda-rank

## Files

- [cuda-rank.cu](cuda-rank.cu)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "hpc.h"

#define BLKDIM 1024

#ifndef SERIAL
__global__ void
rank_kernel( const int *v, int *rank, int n )
{
    __shared__ int other_v[BLKDIM];
    __shared__ int local_rank[BLKDIM];
    const int li = threadIdx.x;
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;

    if (gi >= n)
        return;

    local_rank[li] = 0;

    /* Loop over tiles */
    for (int t=0; t<n; t += blockDim.x) {
        const int this_tile_size = (t + blockDim.x <= n ? blockDim.x : n % blockDim.x);
        /* Fetch an element and populate the tile; make sure we don't
           fetch outside the array bound. */
        if (li < this_tile_size)
            other_v[li] = v[t + li];
        __syncthreads();
        /* compare v[gi] to other_v[]; gj is the global index
           corresponding to the local index lj. */
        for (int lj=0, gj = t; lj<this_tile_size; lj++, gj++) {
            if ( (v[gi] > other_v[lj]) || (v[gi] == other_v[lj] && gi < gj) )
                local_rank[li]++;
        }
        __syncthreads();
    }

    /* Update ranks */
    rank[gi] = local_rank[li];
}
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
    int *d_v, *d_r;
    const size_t SIZE = n * sizeof(int);
    cudaSafeCall( cudaMalloc((void**)&d_v, SIZE) );
    cudaSafeCall( cudaMalloc((void**)&d_r, SIZE) );
    cudaSafeCall( cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice) );
    rank_kernel<<< (n + BLKDIM-1)/BLKDIM, BLKDIM >>>(d_v, d_r, n);
    cudaSafeCall( cudaMemcpy(r, d_r, SIZE, cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaFree(d_v) );
    cudaSafeCall( cudaFree(d_r) );
#endif
}

int main( int argc, char *argv[])
{
    int *v = NULL, *r = NULL;
    int n = 100*BLKDIM;

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

    return EXIT_SUCCESS;
}
