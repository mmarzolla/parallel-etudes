/****************************************************************************
 *
 * omp-bsearch.c - Generalized binary search
 *
 * Copyright (C) 2022, 2024 Moreno Marzolla
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
% Generalized binary search
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-11-15

Implementation of CREW Search, p. 115 of Selim G. Akl, _The Design and
Analysis of Parallel Algorithms_, Prentice-Hall International Editions, 1989,
ISBN 0-13-200073-3

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-bsearch.c -o omp-bsearch

Run with:

        ./omp-bsearch [len [key]]

Example:

        ./omp-bsearch

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "hpc.h"

void vec_init( int *x, int n )
{
    for (int i=0; i<n; i++) {
        x[i] = i;
    }
}

typedef int (*bsearch_fun_t)(const int *, int, int);

int seq_bsearch(const int *x, int n, int key)
{
    int lower = 0, upper = n-1;
    while (lower <= upper) {
        const int m = (upper + lower) / 2;
        if (x[m] == key)
            return m;
        else if (x[m] < key)
            lower = m+1;
        else
            upper = m-1;
    }
    return -1; /* not found */
}

/* Returns the index of one occurrence of `key` in the sorted array
   `x[]` of length `n >= 0`. A negative result means that the key is
   not present. */
int omp_bsearch(const int *x, int n, int key)
{
    const int P = omp_get_max_threads();
    int cmp[P];
    int m[P];
    int start = 0, end = n-1;
#pragma omp parallel default(none) shared(start, end, cmp, m, x, key, P)
    {
        const int my_id = omp_get_thread_num();

        while (end-start > P) {
            m[my_id] = start + ((end-start)*my_id + P)/(P+1);
            cmp[my_id] = (x[m[my_id]] < key ? 1 : -1);
#pragma omp barrier
            /* Assertion:

               cmp[i] == 1 -> key position is > m[i], if present
               cmp[i] == -1 -> key position is <= m[i], if present

               Note that the conditions are such that there is no race
               condition updating `end` and `start`, since only one
               thread (the one with the "correct" result) will update
               them. */
            if (my_id == 0 && cmp[my_id] < 0)
                end = m[my_id];
            else if (my_id == P-1 && cmp[my_id] > 0)
                start = m[my_id-1]+1;
            else if (my_id > 0 && cmp[my_id-1] > 0 && cmp[my_id] < 0) {
                start = m[my_id-1]+1;
                end = m[my_id];
            }
#pragma omp barrier
        } // while
    } // parallel

    for (int i=start; i<=end; i++) {
        if (x[i] == key)
            return i;
    }
    return -1;
}

int randab(int a, int b)
{
    return a + (rand() / 1000) % (b-a+1);
}

void test( bsearch_fun_t f, const int *x, int n )
{
    const int K = 10000; /* number of search keys */
    const int R = 100; /* number of repetitions */
    int *keys = (int*)malloc(K * sizeof(*keys)); assert(keys != NULL);

    for (int i=0; i<K; i++) {
        keys[i] = randab(-1, n+1);
    }

    const double tstart = hpc_gettime();
    for (int r = 0; r<R; r++) {
        for (int i = 0; i<K; i++) {
            const int key = keys[i];
            // printf("Searching for %d on %d elements... ", key, n);
            const int result = f(x, n, key);
            // printf("result=%d\n", result);
            const int expected = (key < 0 || key >= n ? -1 : key);
            assert(result == expected);
        }
    }
    const double elapsed = K * R / (hpc_gettime()-tstart) / 1e6;
    printf("%f Msearches/s\n", elapsed);
}


int main( int argc, char* argv[] )
{
    int n = 1024*1024;
    int *x;
    const int max_len = n * 1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_len ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", max_len);
        return EXIT_FAILURE;
    }

    const size_t size = n * sizeof(*x);
    x = (int*)malloc(size);
    vec_init(x, n);
    printf("\nParallel binary search:     "); fflush(stdout);
    test(omp_bsearch, x, n);
    vec_init(x, n); /* to prevent cache reuse */
    printf("\nSequential binary search:   "); fflush(stdout);
    test(seq_bsearch, x, n);

    /* Cleanup */
    free(x);
    return EXIT_SUCCESS;
}
