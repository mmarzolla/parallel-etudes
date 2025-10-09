/****************************************************************************
 *
 * opencl-sieve.c - Sieve of Eratosthenes
 *
 * Copyright (C) 2018--2025 Moreno Marzolla
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
% Sieve of Eratosthenes
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

## Files

- [opencl-sieve.c](opencl-sieve.c)
- [simpleCL.h](simpleCL.h) [simpleCL.c](simpleCL.c)
- [hpc.h](hpc.h)

***/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

int primes(int n)
{
    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);
    int nprimes = n-2;

    /* Initially, all numbers are considered primes */
    for (int i=0; i<=n; i++)
        isprime[i] = 1;

    sclKernel mark_kernel = sclCreateKernel("mark_kernel");
#ifdef USE_REDUCE
    sclKernel next_prime_kernel = sclCreateKernel("next_prime_kernel_reduce");
    const sclDim GRID_next = DIM1(1);
    const sclDim BLOCK_next = DIM1(SCL_DEFAULT_WG_SIZE1D);
#else
    sclKernel next_prime_kernel = sclCreateKernel("next_prime_kernel");
    const sclDim GRID_next = DIM1(1);
    const sclDim BLOCK_next = DIM1(1);
#endif

    cl_mem d_isprime = sclMallocCopy(n+1, isprime, CL_MEM_READ_WRITE);
    cl_mem d_nprimes = sclMallocCopy(sizeof(nprimes), &nprimes, CL_MEM_READ_WRITE);

    /* main iteration of the sieve */
    int k = 2;
    cl_mem d_next_prime = sclMalloc(sizeof(k), CL_MEM_WRITE_ONLY);
    /* promote k to `long` to avoid overflow; promote `n` to `long` to
       avoid compiler warning */
    while (((long)k)*k <= (long)n) {
        const int from = k*k; /* here we know that k*k does not overflow */
        const int to = n;
        const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE1D);
        const sclDim GRID = DIM1(sclRoundUp((to - from + k-1)/k, SCL_DEFAULT_WG_SIZE1D));

        sclSetArgsEnqueueKernel(mark_kernel,
                                GRID, BLOCK,
                                ":b :d :d :d :b :L",
                                d_isprime, k, from, to, d_nprimes, BLOCK.sizes[0] * sizeof(int));
        sclSetArgsEnqueueKernel(next_prime_kernel,
                                GRID_next, BLOCK_next,
                                ":b :d :d :b",
                                d_isprime, k, n, d_next_prime);
        const int oldk = k;
        sclMemcpyDeviceToHost(&k, d_next_prime, sizeof(k));
        assert(k > oldk);
    }

    sclMemcpyDeviceToHost(&nprimes, d_nprimes, sizeof(nprimes));
    sclFree(d_nprimes);
    sclFree(d_isprime);
    return nprimes;
}

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

    printf("Execution time  %.3f\n", elapsed);

    sclFinalize();

    return EXIT_SUCCESS;
}
