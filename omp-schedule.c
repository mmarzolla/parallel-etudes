/****************************************************************************
 *
 * omp-schedule.c - simulate "schedule()" directives
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Simulate "schedule()" directives
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-06-20

OpenMP allows the use of `schedule(static)` and `schedule(dynamic)`
clauses to assign iterations of a "for" loop to OpenMP threads. The
purpose of this exercise is to simulate these clauses _without_ using
the `omp parallel for` construct.

The file [omp-schedule.c](omp-schedule.c) contains a serial program
that performs the following computations. The program creates and
initializes a `vin[]` array of length $n$ (the value $n$ can be passed
from the command line). The program creates a second array `vout[]` of
length $n$, and defines its content such that `vout[i] = Fib(vin[i])`
for each $i$, where `Fib(k)` the _k_th number of the Fibonacci
sequence: `Fib(0)` = `Fib(1)` = 1; `Fib(k) = Fib(k-1) + Fib(k-2)` if
$k \geq $2. To compute `fib(k)` we deliberately use the inefficient
recursive algorithm, so that the computation time strongly depends on
_k_.

There are two identical functions, `do_static()` and `do_dynamic()`
that perform the computation above.

1. Modify `do_static()` to distribute the loop iterations to OpenMP
   threads as would be done by the `schedule(static, chunk_size)`
   directive, but without using an explicit `omp parallel for`
   construct (you may use `omp parallel`).

2. Then, modify `do_dynamic()` to distribute the loop iterations to
   OpenMP threads as would be done by the `schedule(dynamic,
   chunk_size)` directive. Again, you are not allowed to use `omp
   parallel for`, but only `omp parallel`.

**Suggestion.** Start assuming `chunk_size = 1`. When implementing
dynamic schedule (point 2 above) I suggest to assume `chunk_size = 1`,
and use a shared variable to keep track of the index of the next
element of `vin[]` which must be processed. Each thread gets the next
unprocessed elements from `vin[]` and increments the shared variable
atomically. This is tricky and must be done carefully.

If you have time, compare the execution times of your implementation
with those obtained by using the `omp parallel for` directive with the
appropriate `schedule()` clauses.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-schedule.c -o omp-schedule

To execute:

        ./omp-schedule [n]

Example:

        OMP_NUM_THREADS=2 ./omp-schedule

## Files

- [omp-schedule.c](omp-schedule.c)

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

/* Fill vectors `vin` and `vout` of length `n`; `vin` will contain
   input values; `vout` is initialized with -1 */
void fill(int *vin, int *vout, int n)
{
    int i;
    /* fill input array */
    for (i=0; i<n; i++) {
        vin[i] = 25 + (i%10);
        vout[i] = -1;
    }
}

/* Check correctness of `vout[]`. Return 1 if correct, 0 if not */
int check(const int *vin, const int *vout, int n)
{
    int i;
    /* check result */
    for (i=0; i<n; i++) {
        if ( vout[i] != fib_iter(vin[i]) ) {
            fprintf(stderr,
                    "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n",
                    i, vin[i], i, vout[i], fib_iter(vin[i]));
            return 0;
        }
    }
    fprintf(stderr, "Test OK\n");
    return 1;
}


void do_static(const int *vin, int *vout, int n)
{
    int i;
#ifdef SERIAL
    /* [TODO] parallelize the following loop, simulating a
       "schedule(static,1)" clause, i.e., static scheduling with block
       size 1. Do not modify the body of the fib_rec() function. */
    for (i=0; i<n; i++) {
        vout[i] = fib_rec(vin[i]);
        /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
    }
#else
    const int chunk_size = 1; /* can be set to any value >= 1 */
#pragma omp parallel default(none) shared(vin,vout,n,chunk_size) private(i)
    {
        /* This implementation simulates the behavior of a
           schedule(static,chunk_size) clause for any chunk_size>=1. */
        const int my_id = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        const int START = my_id * chunk_size;
        const int STRIDE = n_threads * chunk_size;
        int j;

        for (i=START; i<n; i += STRIDE) {
            for (j=i; j<i+chunk_size && j<n; j++) {
                vout[j] = fib_rec(vin[j]);
                /* printf("Thread %d vin[%d]=%d vout[%d]=%d\n", tid, j, vin[j], j, vout[j]); */
            }
        }
    }
#endif
}

void do_dynamic(const int *vin, int *vout, int n)
{
    int i;
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
#pragma omp parallel default(none) shared(idx,vin,vout,n,chunk_size) private(i)
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
            for (i=my_idx; i<my_idx+chunk_size && i<n; i++) {
                vout[i] = fib_rec(vin[i]);;
                /* printf("Thread %d vin[%d]=%d vout[%d]=%d\n", tid, i, vin[j], j, vout[j]); */
            }
        } while (my_idx < n);
    }
#endif
}

int main( int argc, char* argv[] )
{
    int n = 1024;
    const int max_n = 512*1024*1024;
    int *vin, *vout;
    double tstart, elapsed;

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

    /**
     ** First test
     **/
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_static(vin, vout, n);
    elapsed = omp_get_wtime() - tstart;
    check(vin, vout, n);

    printf("Elapsed time (static schedule): %f\n", elapsed);

    /**
     ** First test
     **/
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_dynamic(vin, vout, n);
    elapsed = omp_get_wtime() - tstart;
    check(vin, vout, n);

    printf("Elapsed time (dynamic schedule): %f\n", elapsed);

    free(vin);
    free(vout);
    return EXIT_SUCCESS;
}
