/****************************************************************************
 *
 * omp-sum.c - Sum-reduction of an array
 *
 * Copyright (C) 2023 by Alice Girolomini <alice.girolomini(at)studio.unibo.it>
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
% HPC - Sum-reduction of an array
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-10-27

The file [omp-sum.c](omp-sum.c) contains a serial implementation of an
OpenMP program that computes the sum of an array of length $N$; indeed,
the program performs a _sum-reduction_ of the array. 

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-sum.c -o omp-sum

To execute:

        ./omp-sum N

Example:

        ./omp-sum 1000000

## Files

- [omp-sum.c](omp-sum.c)

***/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef SERIAL
/** 
 * Computes the sum of all elements of array `v` of length `n` 
*/
float sum (float *v, int n) {
    float sum = 0;
    int i;

    for (i = 0; i < n; i++) {
        sum += v[i];
    }

    return sum;
}

/**
 * Fills the array `v` of length `n`; returns the sum of the
 * content of `v`
*/
float fill (float *v, int n) {
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);
    int i;

    for (i = 0; i < n; i++) {
        v[i] = vals[i % NVALS];
    }
    
    switch(i % NVALS) {
    case 1: return 1; break;
    case 3: return 2; break;
    default: return 0;
    }
}

#else
/**
 * Each thread computes the local sum then performs a reduction
 */
float sum (float *v, int n) {
    float sum = 0;
    int i;

    #pragma omp parallel for default(none) shared(n, v) reduction(+ : sum)
    for (i = 0; i < n; i++) {
        sum += v[i];
    }

    return sum;
}

/** 
 * Each thread fills its portion of the array `v` of length `n`; 
 * returns the expected result of the reduction
*/
float fill (float *v, int n) {
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);
    int i;

    #if __GNUC__ < 9 
    #pragma omp parallel for default(none) shared(n, v)
    #else
    #pragma omp parallel for default(none) shared(n, vals, NVALS, v)
    #endif
    for (i = 0; i < n; i++) {
        v[i] = vals[i % NVALS];
    }

    switch(n % NVALS) {
    case 1: return 1; break;
    case 3: return 2; break;
    default: return 0;
    }
}
#endif

int main (int argc, char *argv[]) {
    size_t n = 10000; 
    float s, expected;
    float *v;

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    v = (float*)malloc( n * sizeof(float) );
    assert(v != NULL);

    const double tstart = omp_get_wtime();
    expected = fill(v, n);
    s = sum(v, n);
    const double elapsed = omp_get_wtime() - tstart;
    fprintf(stderr,"\nExecution time %f seconds\n", elapsed);

    printf("Sum=%f, expected=%f\n", s, expected);
    if (s == expected) {
        printf("Test OK\n");
    } else {
        printf("Test FAILED\n");
    }
    
    free(v);
    return EXIT_SUCCESS;
}