/****************************************************************************
 *
 * omp-dot.c - Dot product
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
% Dot product
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

The file [omp-dot.c](omp-dot.c) contains a serial program that
computes the dot product of two arrays `v1[]` and `v2[]`. The program
receives the array lengths $n$ as the only parameter on the command
line. The arrays are initialized deterministically, so that their
scalar product is known awithout computing it explicitly.  The dot
product of `v1[]` and `v2[]` is defined as:

$$
\sum_{i = 0}^{n-1} v1[i] \times v2[i]
$$

The goal of this exercise is to parallelize the serial program using
the `omp parallel` construct with the appropriate clauses. It is
instructive to begin without using the `omp parallel for` directive
and computing the endpoints of the iterations explicitly.  To this
aim, let $P$ be the size of the OpenMP team; partition the
arrays into $P$ blocks of approximately uniform size. Thread $p$ ($0
\leq p < P$) computes the dot product `my_p` of the subvectors with
indices $\texttt{my_start}, \ldots, \texttt {my_end}-1$:

$$
\texttt{my_p}: = \sum_{i=\texttt{my_start}}^{\texttt{my_end}-1} v1[i] \times v2[i]
$$

There are several ways to accumulate partial results. One possibility
is to store the value computed by thread $p$ on `partial_p[p]`, where
`partial_p[]` is an array of length $P$. In this way each thread
writes on different elements of `partial_p[]` and no race conditions
are possible. After the parallel region completes, the master thread
computes the final result by summing the content of `partial_p[]`. Be
sure to handle the case where $n$ is not an integer multiple of $P$
correctly.

The solution above is instructive but tedious and inefficient.  Unless
there are specific reasons to do so, in practice you should use the
`omp parallel for` directive with the `reduction()` clause, and let
the compiler take care of everything.

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
    for (size_t i=0; i<n; i++) {
        v1[i] = seq1[i%3];
        v2[i] = seq2[i%3];
    }
}

int dot(const int *v1, const int *v2, size_t n)
{
#ifdef SERIAL
    /* [TODO] Parallelize the following loop */
    int result = 0;
    for (size_t i=0; i<n; i++) {
        result += v1[i] * v2[i];
    }
#else
#if 0
    /* This version uses neither "parallel for" nor "reduction"
       directives; although this solution should not be used in
       practice, it is instructive to try it. */
    const int P = omp_get_max_threads();
    int partial_p[P];
#pragma omp parallel default(none) shared(P, v1, v2, n, partial_p)
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
#else
    /* This is the efficient solution that relies on the "parallel
       for" and "reduction" directives */
    int result = 0;
#pragma omp parallel for default(none) shared(v1, v2, n) reduction(+:result)
    for (size_t i=0; i<n; i++) {
        result += v1[i] * v2[i];
    }
#endif
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
        fprintf(stderr, "FATAL: Array too long (requested length=%lu, maximum length=%lu\n", (unsigned long)n, (unsigned long)n_max);
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
    printf("Execution time %.3f\n", elapsed);
    free(v1);
    free(v2);

    return EXIT_SUCCESS;
}
