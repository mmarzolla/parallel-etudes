/****************************************************************************
 *
 * omp-dot.c - Dot product
 *
 * Copyright (C) 2018--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Dot product
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-09-14

The file [omp-dot.c](omp-dot.c) contains a serial program that
computes the dot product of two arrays `v1[]` and `v2[]`. The program
accepts the array lengths $n$ as the only command line parameter. The
arrays are initialized deterministically in order to know their scalar
product without the need to compute it; this is useful for
testing.. Recall that the dot product of `v1[]` and `v2[]` is
defined as:

$$
\sum_{i = 0}^{n-1} v1[i] \times v2[i]
$$

Parallelize the serial program using the `omp parallel` construct with
the appropriate clauses. It is instructive to begin without using the
`omp parallel for` directive, and calculating the endpoints of the
iterations by hand as follows: let $P$ be the size of the OpenMP
thread pool, partition the arrays into $P$ blocks of approximately
uniform size. Thread $p$ ($0 \leq p < P$) computes the dot product
`my_p` of the subvectors with indices $\texttt{my_start}, \ldots,
\texttt {my_end}-1$:

$$
\texttt{my_p}: = \sum_{i=\texttt{my_start}}^{\texttt{my_end}-1} v1[i] \times v2[i]
$$

There are several ways to accumulate partial results. One possibility
is to make sure that the value calculated by thread $p$ is stored in
`partial_p[p]`, where `partial_p[]` is an array of length $P$; in this
way each thread handles a different element of `partial_p[]` and does
not check _race condition_. The master computes the final result as
the sum of the values ​​in `partial_p[]`. Be careful to manage correctly
the case where the length $n$ of the arrays is not a multiple of $P$.


The solution above is instructive but very tedious. In fact, unless
there are specific reasons to do otherwise, you should use the `omp
parallel for` directive with the `reduction()` clause, and let the
compiler take care of everything. 

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-dot.c -o omp-dot

To execute:

        ./omp-dot [n]

For example, if you want to use two OpenMP threads:

        OMP_NUM_THREADS=2 ./omp-dot 1000000

## File

- [omp-dot.c](omp-dot.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

void fill( int *v1, int *v2, size_t n )
{
    const int seq1[3] = { 3, 7, 18};
    const int seq2[3] = {12, 0, -2};
    size_t i;
    for (i=0; i<n; i++) {
        v1[i] = seq1[i%3];
        v2[i] = seq2[i%3];
    }
}

int dot(const int *v1, const int *v2, size_t n)
{
#ifdef SERIAL
    /* [TODO] Parallelize the following loop */
    result = 0;
    for (int i=0; i<n; i++) {
        result += v1[i] * v2[i];
    }
#else
    /* The serial code should be parallelized using

#pragma omp parallel for default(none) shared(v1,v2,n) reduction(+:dotprod)

       since we have not seen the "for" clause so far, we
       parallelize the loop "by hand". */
    const int P = omp_get_max_threads();
    int partial_p[P];
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(v1, v2, n, partial_p)
#else
#pragma omp parallel default(none) shared(P, v1, v2, n, partial_p)
#endif
    {
        const int my_id = omp_get_thread_num();
        const size_t my_start = (n * my_id) / P;
        const size_t my_end = (n * (my_id + 1)) / P;
        int my_p = 0;
        /* printf("Thread %d P=%d my_start=%lu my_end=%lu\n", my_id, P, (unsigned long)my_start, (unsigned long)my_end); */
        for (size_t j=my_start; j<my_end; j++) {
            my_p += v1[j] * v2[j];
        }
        partial_p[my_id] = my_p;
        /* printf("partial_sum[%d]=%d\n", my_id, partial_sum[my_id]); */
    } /* implicit barrier here */

    /* we are outside a parallel region, so what follows is done by
       the master only */
    int result = 0;
    for (int i=0; i<P; i++) {
        result += partial_p[i];
    }
#endif
    return result;
}

int main( int argc, char *argv[] )
{
    size_t n = 10*1024*1024l; /* array length */
    const size_t n_max = 512*1024*1024l; /* max length */
    int *v1, *v2;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atol(argv[1]);
    }

    if ( n > n_max ) {
        fprintf(stderr, "FATAL: The array length must be at most %lu\n", (unsigned long)n_max);
        return EXIT_FAILURE;
    }

    printf("Initializing array of length %lu\n", (unsigned long)n);
    v1 = (int*)malloc( n*sizeof(v1[0])); assert(v1 != NULL);
    v2 = (int*)malloc( n*sizeof(v2[0])); assert(v2 != NULL);
    fill(v1, v2, n);

    const int expect = (n % 3 == 0 ? 0 : 36);

    const double tstart = omp_get_wtime();
    const int result = dot(v1, v2, n);
    const double elapsed = omp_get_wtime() - tstart;

    if ( result == expect ) {
        printf("Test OK\n");
    } else {
        printf("Test FAILED: expected %d, got %d\n", expect, result);
    }
    printf("Elapsed time: %f\n", elapsed);
    free(v1);
    free(v2);

    return EXIT_SUCCESS;
}
