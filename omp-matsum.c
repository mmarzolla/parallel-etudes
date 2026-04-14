/****************************************************************************
 *
 * omp-matsum.c - Matrix-matrix addition
 *
 * Copyright (C) 2026 Moreno Marzolla
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
% Matrix-matrix addition
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-14

The program [omp-matsum.cc](omp-matsum.c) computes the sum of two
square matrices of size $N \times N$. Modify the program to use OpenMP
parallelism.

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-matsum.c -o omp-matsum

To execute:

        OMP_NUM_THREADS=[P] ./omp-matsum [N]

Example:

        OMP_NUM_THREADS=2 ./omp-matsum 1024

## Files

- [omp-matsum.c](omp-matsum.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

/* Compute the sum of `p` and `q`; store the result in `r`. All
   parameters are nxn square matrices. */
void matsum( const float *p, const float *q, float *r, int n )
{
#ifndef SERIAL
#pragma omp parallel for default(none) shared(p,q,r,n) schedule(static)
#endif
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            r[i*n + j] = p[i*n + j] + q[i*n + j];
        }
    }
}

/* Initialize square matrix p of size nxn. */
void fill( float *p, int n )
{
    int k = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            p[i*n+j] = k;
            k = (k+1) % 1000;
        }
    }
}

/* Check result. */
int check( float *r, int n )
{
    int k = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (fabs(r[i*n+j] - 2.0f*k) > 1e-5) {
                fprintf(stderr, "Check FAILED: r[%d][%d] = %f, expeted %f\n", i, j, r[i*n+j], 2.0*k);
                return 0;
            }
            k = (k+1) % 1000;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    float *p, *q, *r;
    int n = 1024;
    const int max_n = 5000;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: the maximum allowed matrix size is %d\n", max_n);
        return EXIT_FAILURE;
    }

    printf("Matrix size: %d x %d\n", n, n);

    const size_t size = n*n*sizeof(*p);

    /* Allocate space for p, q, r */
    p = (float*)malloc(size); assert(p != NULL);
    fill(p, n);
    q = (float*)malloc(size); assert(q != NULL);
    fill(q, n);
    r = (float*)malloc(size); assert(r != NULL);

    const double tstart = omp_get_wtime();
    matsum(p, q, r, n);
    const double elapsed = omp_get_wtime() - tstart;

    printf("Execution time (s): %.3f\n", elapsed);
    printf("Throughput (Melements/s): %.3f\n", n*n/(1e6 * elapsed));

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
