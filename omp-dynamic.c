/****************************************************************************
 *
 * omp-dynamic.c - simulate "schedule(dynamic)" using the "omp parallel" clause
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Implementing the "schedule(dynamic)" clause by hand
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-18

We know that the `schedule(dynamic)` OpenMP clause dynamically assigns
iterations of a "for" loop to the first available OpenMP thread. The
purpose of this exercise is to simulate the `schedule(dynamic)` clause
using only the `omp parallel` construct (not` omp parallel for`).

File [omp-dynamic.c](omp-dynamic.c) contains a serial program that
initializes an array `vin[]` with $n$ random integers (the value of
$n$ can be passed from the command line). The program creates a second
array `vout[]` of the same length, where `vout[i] = Fib(vin[i])` for
each $i$. `Fib(k)` is the _k_-th Fibonacci number: `Fib(0) = Fib(1) =
1`; `Fib(k) = Fib(k-1) + Fib(k-2)` for $k \geq 2$. The computation of
Fibonacci numbers is deliberately inefficient to ensure that there are
huge variations of the running time depending on the argument.

First, try to parallelize the "for" loop indicated by the `[TODO]`
comment using the `omp parallel for` construct. Keeping the array
length constant, measure the execution times in the following cases:

1. Using the `#pragma omp parallel for` directive with the default
   schedule, which for GCC is the static schedule with chunk size
   equal to $n / \textit{thread_pool_size}$);

2. Using the `#pragma omp parallle for schedule(static,..)` with a
   smaller chunk size, e.g. 64 or less;

3. Using the `#pragma omp parallel for schedule(dybamic)` directive;
   recall that in this case the default chunk size is 1.

Then, try to simulate the same behavior of step 3 (dynamic schedule
with chunk size set to 1) using the `omp parallel` construct
only. Proceed as follows: inside the parallel region, use a shared
variable `idx` to hold the index of the next element of `vin[]` that
must be processed. Each thread atomically acquires the current value
of `idx` and then increments the shared value. This operation should
be carefully performed inside a critical region. However, the actual
computation of `vin[]` must be performed _outside_ the critical
region, so that the threads can actually compute in parallel.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-dynamic.c -o omp-dynamic

To execute:

        ./omp-dynamic [n]

Example:

        OMP_NUM_THREADS=2 ./omp-dynamic

## Files

- [omp-dynamic.c](omp-dynamic.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Recursive computation of the n-th Fibonacci number, for n=0, 1, 2, ...
   Do not parallelize this function. */
int fib_rec(int n)
{
    if (n<2) {
        return 1;
    } else {
        return fib_rec(n-1) + fib_rec(n-2);
    }
}

/* Iterative computation of the n-th Fibonacci number. This function
   must be used for checking the result only. */
int fib_iter(int n)
{
    if (n<2) {
        return 1;
    } else {
        int fibnm1 = 1;
        int fibnm2 = 1;
        int fibn;
        n = n-1;
        do {
            fibn = fibnm1 + fibnm2;
            fibnm2 = fibnm1;
            fibnm1 = fibn;
            n--;
        } while (n>0);
        return fibn;
    }
}

/* Initialize the content of vector v using the values from vstart to
   vend.  The vector is filled in such a way that there are more or
   less the same number of contiguous occurrences of all values in
   [vstart, vend]. */
void fill(int *v, int n)
{
    const int vstart = 20, vend = 35;
    const int blk = (n + vend - vstart) / (vend - vstart + 1);
    int tmp = vstart;

    for (int i=0; i<n; i+=blk) {
        for (int j=0; j<blk && i+j<n; j++) {
            v[i+j] = tmp;
        }
        tmp++;
    }
}

int main( int argc, char* argv[] )
{
    int i, n = 1024;
    const int max_n = 512*1024*1024;
    int *vin, *vout;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    /* initialize the input and output arrays */
    vin = (int*)malloc(n * sizeof(vin[0])); assert(vin != NULL);
    vout = (int*)malloc(n * sizeof(vout[0])); assert(vout != NULL);

    /* fill input array */
    for (i=0; i<n; i++) {
        vin[i] = 25 + (i%10);
    }

    const double tstart = omp_get_wtime();

#ifdef SERIAL
    /* [TODO] parallelize the following loop, simulating a
       "schedule(dynamic,1)" clause, i.e., dynamic scheduling with
       block size 1. Do not modify the body of the fib_rec()
       function. */
    for (i=0; i<n; i++) {
        vout[i] = fib_rec(vin[i]);
        /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
    }
#else
    int idx = 0; /* shared index */
    const int chunk_size = 1; /* can be set to any value >= 1 */
#pragma omp parallel default(none) shared(idx,vin,vout,n,chunk_size)
    {
        /* This implementation simulates the behavior of a
           schedule(dynamic,chunk_size) clause for any chunk_size>=1. */
        int my_idx;
        do {
            /* atomically grab current index, and increment */
#pragma omp critical
            {
                my_idx = idx;
                idx += chunk_size;
            }
            for (int j=my_idx; j<my_idx+chunk_size && j<n; j++) {
                vout[j] = fib_rec(vin[j]);;
                /* printf("Thread %d vin[%d]=%d vout[%d]=%d\n", tid, j, vin[j], j, vout[j]); */
            }
        } while (my_idx < n);
    }
#endif

    const double elapsed = omp_get_wtime() - tstart;

    /* check result */
    for (i=0; i<n; i++) {
        if ( vout[i] != fib_iter(vin[i]) ) {
            fprintf(stderr,
                    "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n",
                    i, vin[i], i, vout[i], fib_iter(vin[i]));
            return EXIT_FAILURE;
        }
    }
    printf("Test OK\n");
    printf("Elapsed time: %f\n", elapsed);

    free(vin);
    free(vout);
    return EXIT_SUCCESS;
}
