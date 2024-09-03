/****************************************************************************
 *
 * cuda-sieve.c -- Sieve of Eratosthenes
 *
 * Copyright (C) 2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Sieve of Eratosthenes
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-09-02

## Files

- [cuda-sieve.c](cuda-sieve.cu)
- [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#ifndef SERIAL
#define BLKDIM 32

/**
 * Mark all multiples of k belonging to the set {from, ... to-1}.
 * from must be a multiple of k. The number of elements that are
 * marked for the first time is atomically subtracted from *nprimes.
 */
__kernel void
mark_kernel( char *isprime,
             int k,
             int from,
             int to,
             int *nprimes,
             char *mark)
{
    const int i = from + (threadIdx.x + blockIdx.x * blockDim.x)*k;
    const int li = threadIdx.x;

    mark[li] = 0;
    __syncthreads();

    if (i < to) {
        mark[li] = (isprime[i] == 1);
        isprime[i] = 0;
    }

    __syncthreads();

    int d = blockDim.x;
    while (d > 1) {
        int d2 = (d + 1)/2;
        if (li + d2 < d) mark[li] += mark[li + d2];
        d = d2;
        __syncthreads();
    }
    if (0 == li) {
        atomic_sub(nprimes, mark[0]);
    }
}

__kernel void
next_prime_kernel(const char *isprime,
                  int k,
                  int n,
                  int *next_prime)
{
    if (threadIdx.x == 0) {
        k++;
        while (k < n && isprime[k] == 0)
            k++;
        *next_prime = k;
    }
}
#endif

int primes(int n)
{
    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);
    int nprimes = n-2;

    /* Initially, all numbers are considered primes */
    for (int i=0; i<=n; i++)
        isprime[i] = 1;

    char *d_isprime;
    ing *d_nprimes, *d_next_prime;

    cudaSafeCall( cudaMalloc( (void**)&d_isprime, n+1) );
    cudaSafeCall( cudaMalloc( (void**)&d_nprimes, sizeof(*d_nprimes)) );
    cudaSafeCall( cudaMalloc( (void**)&d_next_prime, sizeof(*d_next_prime)) );

    const dim3 BLOCK(BLKSIZE);
    const dim3 GRID((to - from + k-1)/k);
    /* main iteration of the sieve */
    int k = 2;
    while (k*k <= n) {
        const int from = k*k;
        const int to = n;
        const int nelem = (to - from + k-1)/k;
        const dim3 GRID((nelem + BLKDIM - 1)/BLKDIM);
        mark_kernel<<<GRID, BLOCK>>>(d_isprime, k, from, to, d_nprimes, d_isprime);
        next_prime_kernel<<<1, 1>>>(d_isprime, k, n, d_next_prime);
        const int oldk = k;
        cudaSafeCall( cudaMemcpy(&k, d_next_prime, sizeof(k), cudaMemcpyDeviceToHost) );
        assert(k > oldk);
    }
    cudaSafeCall( cudaMemcpy(&nprimes, d_nprimes, sizeof(nprimes), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaFree(d_nprimes) );
    cudaSafeCall( cudaFree(d_isprime) );
    return nprimes;
}
#endif

int main( int argc, char *argv[] )
{
    int n = 1000000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc == 2 ) {
        n = atol(argv[1]);
    }

    sclInitFromFile("opencl-sieve.cl");
    const double tstart = hpc_gettime();
    const int nprimes = primes(n);
    const double elapsed = hpc_gettime() - tstart;

    printf("There are %d primes in {2, ..., %d}\n", nprimes, n);

    printf("Elapsed time: %f\n", elapsed);

    sclFinalize();

    return EXIT_SUCCESS;
}
