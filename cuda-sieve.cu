/****************************************************************************
 *
 * cuda-sieve.c - Sieve of Eratosthenes
 *
 * Copyright (C) 2024, 2025 Moreno Marzolla
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
% HPC - Sieve of Eratosthenes
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

![Eratosthenes (276 BC--194 BC).](Eratosthenes.png "Etching of an ancient seal identified as Eartosthenes")

The _sieve of Erathostenes_ is an algorithm for identifying the prime
numbers within the set $\{2, \ldots, n\}$. An integer $p \geq 2$ is
prime if and only if its only divisors are 1 and $p$ itself (2 is
prime).

To illustrate how the sieve of Eratosthenes works, let us consider
$n=20$. We start by listing all integers $2, \ldots n$:

![](sieve1.svg)

The first value in the list (2) is prime; we mark all its multiples,
and get:

![](sieve2.svg)

The next unmarked value (3) is prime. We mark all its multiples
starting from $3 \times 3$, since $3 \times 2$ has already been marked
as a multiple of two. We get:

![](sieve3.svg)

The next unmarked value (5) is prime. The smaller unmarked multiple of
5 is $5 \times 5$, because $5 \times 2$, $5 \times 3$ and $5 \times 4$
have already been marked as multiples of 2 and 3. However, since $5
\times 5$ is outside the upper bound of the interval, the algorithm
terminates and all unmarked numbers are prime:

![](sieve4.svg)

The file [cuda-sieve.cu](cuda-sieve.cu) contains a serial program that
takes as input an integer $n \geq 2$, and computes the number $\pi(n)$
of primes in the set $\{2, \ldots n\}$ using the sieve of
Eratosthenes[^1]. Although the serial program could be made more
efficient, for the sake of this exercise we trade efficiency for
readability.

The set of unmarked numbers in $\{2, \ldots, n\}$ is represented by
the `isprime[]` array of length $n+1$; during execution, `isprime[k]`
is 0 if and only if $k$ has been marked, i.e., has been determined to
be composite; `isprime[0]` and `isprime[1]` are not used.

[^1]: $\pi(n)$ is the [prime-counting
      function](https://en.wikipedia.org/wiki/Prime-counting_function)

The goal of this exercise is to write a parallel version of the sieve
of Erathostenes using CUDA.

## Files

- [cuda-sieve.c](cuda-sieve.cu)
- [hpc.h](hpc.h)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

#ifdef SERIAL
/* Mark all mutliples of `k` in the set {`from`, ..., `to`-1}; return
   how many numbers have been marked for the first time. `from` does
   not need to be a multiple of `k`, although in this program it
   always is. */
int mark( char *isprime, int k, int from, int to )
{
    int nmarked = 0;
    from = ((from + k - 1)/k)*k; /* start from the lowest multiple of p that is >= from */
    for ( int x=from; x<to; x+=k ) {
        if (isprime[x]) {
            isprime[x] = 0;
            nmarked++;
        }
    }
    return nmarked;
}
#else
#define BLKDIM 1024

/**
 * Mark all multiples of k belonging to the set {from, ... to-1}.
 * from must be a multiple of k. The number of elements that are
 * marked for the first time is atomically subtracted from *nprimes.
 */
__global__ void
mark_kernel( char *isprime,
             int k,
             int from,
             int to,
             int *nprimes )
{
    __shared__ int mark[BLKDIM];
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
        atomicSub(nprimes, mark[0]);
    }
}

__global__ void
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

#ifdef SERIAL
    /* main iteration of the sieve */
    for (int i=2; ((long)i)*i <= (long)n; i++) {
        if (isprime[i]) {
            nprimes -= mark(isprime, i, i*i, n+1);
        }
    }
#else
    char *d_isprime;
    int *d_nprimes, *d_next_prime;

    cudaSafeCall( cudaMalloc( (void**)&d_isprime, n+1) );
    cudaSafeCall( cudaMemcpy( d_isprime, isprime, n+1, cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMalloc( (void**)&d_nprimes, sizeof(*d_nprimes)) );
    cudaSafeCall( cudaMemcpy( d_nprimes, &nprimes, sizeof(nprimes), cudaMemcpyHostToDevice) );

    cudaSafeCall( cudaMalloc( (void**)&d_next_prime, sizeof(*d_next_prime)) );

    const dim3 BLOCK(BLKDIM);
    /* main iteration of the sieve */
    int k = 2;
    while (((long)k)*k <= (long)n) {
        const int from = k*k;
        const int to = n;
        const int nelem = (to - from + k-1)/k;
        const dim3 GRID((nelem + BLKDIM - 1)/BLKDIM);
        mark_kernel<<<GRID, BLOCK>>>(d_isprime, k, from, to, d_nprimes); cudaCheckError();
        next_prime_kernel<<<1, 1>>>(d_isprime, k, n, d_next_prime); cudaCheckError();
        const int oldk = k;
        cudaSafeCall( cudaMemcpy(&k, d_next_prime, sizeof(k), cudaMemcpyDeviceToHost) );
        assert(k > oldk);
    }
    cudaSafeCall( cudaMemcpy(&nprimes, d_nprimes, sizeof(nprimes), cudaMemcpyDeviceToHost) );
    cudaSafeCall( cudaFree(d_nprimes) );
    cudaSafeCall( cudaFree(d_isprime) );
#endif
    free(isprime);
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

    const double tstart = hpc_gettime();
    const int nprimes = primes(n);
    const double elapsed = hpc_gettime() - tstart;

    printf("There are %d primes in {2, ..., %d}\n", nprimes, n);

    printf("Execution time %.3f\n", elapsed);

    return EXIT_SUCCESS;
}
