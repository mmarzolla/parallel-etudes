/****************************************************************************
 *
 * opencl-sieve.c -- Sieve of Eratosthenes
 *
 * Copyright (C) 2018--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Last updated: 2023-12-09

## Files

- [opencl-sieve.c](opencl-sieve.c)
- [simpleCL.h](simpleCL.h) [simpleCL.c](simpleCL.c)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

int primes(int n)
{
    char *isprime = (char*)malloc(n+1); assert(isprime != NULL);
    int nprimes = n-2;

    /* Initially, all numbers are considered primes */
    for (int i=0; i<=n; i++)
        isprime[i] = 1;

    sclKernel mark_kernel = sclCreateKernel("mark_kernel");
    sclKernel next_prime_kernel = sclCreateKernel("next_prime_kernel");
    sclKernel next_prime_kernel_reduce = sclCreateKernel("next_prime_kernel_reduce");

    cl_mem d_isprime = sclMallocCopy(n+1, isprime, CL_MEM_READ_WRITE);
    cl_mem d_nprimes = sclMallocCopy(sizeof(nprimes), &nprimes, CL_MEM_READ_WRITE);

    /* main iteration of the sieve */
    int k = 2;
    cl_mem d_next_prime = sclMalloc(sizeof(k), CL_MEM_WRITE_ONLY);
    while (((long)k)*k <= (long)n) {
        const int from = k*k;
        const int to = n;
        const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE1D);
        const sclDim GRID = DIM1(sclRoundUp((to - from + k-1)/k, SCL_DEFAULT_WG_SIZE1D));

        sclSetArgsEnqueueKernel(mark_kernel,
                                GRID, BLOCK,
                                ":b :d :d :d :b :L",
                                d_isprime, k, from, to, d_nprimes, BLOCK.sizes[0] * sizeof(int));
#if 1
        sclSetArgsEnqueueKernel(next_prime_kernel,
                                DIM1(1), DIM1(1),
                                ":b :d :d :b",
                                d_isprime, k, n, d_next_prime);
#else
        sclSetArgsEnqueueKernel(next_prime_kernel_reduce,
                                DIM1(SCL_DEFAULT_WG_SIZE1D), DIM1(SCL_DEFAULT_WG_SIZE1D),
                                ":b :d :d :b",
                                d_isprime, k, n, d_next_prime);
#endif
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

    printf("Elapsed time: %f\n", elapsed);

    sclFinalize();

    return EXIT_SUCCESS;
}
