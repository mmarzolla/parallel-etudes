/****************************************************************************
 *
 * omp-tri-gemv.c - Upper-triangular Matrix-Vector multiply
 *
 * Copyright (C) 2024 Moreno Marzolla
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
% Upper-triangular Matrix-Vector multiply
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-12-04

Given a square matrix $A$ in upper triangular form and a vector $b$,
the function `tri_gemv(A, b, c)` computes $c = Ab$. The goal of this
exercise is to parallelize `tri_gemv()` using OpenMP.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-tri-gemv.c -o omp-tri-gemv -lm

Run with:

        ./omp-tri-gemv [n]

## Files

- [omp-tri-gemv.c](omp-tri-gemv.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void fill(float *A, float *b, int n)
{
    for (int i=0; i<n; i++) {
        b[i] = 1;
        for (int j=0; j<n; j++) {
            A[i*n + j] = (j >= i);
        }
    }
}

void tri_gemv(const float *A, const float *b, float *c, int n)
{
    for (int i=0; i<n; i++) {
        c[i] = 0;
#ifndef SERIAL
#pragma omp parallel for reduction(+:c[i])
#endif
        for (int j=i; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}

void check(const float *c, int n)
{
    static const float EPS = 1e-5;

    for (int i=0; i<n; i++) {
        const float expected = n-i;
        if (fabs(c[i] - expected) > EPS) {
            fprintf(stderr, "c[%d]=%f, expected %f\n", i, c[i], expected);
        }
    }
}

int main( int argc, char *argv[] )
{
    const int MAXN = 20000;
    int n = 100;
    float *A, *b, *c;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAXN) {
        fprintf(stderr, "FATAL: size %d exceeds the maximum %d\n", n, MAXN);
        return EXIT_FAILURE;
    }

    A = (float*)malloc(n*n*sizeof(*A));
    b = (float*)malloc(n*sizeof(*b));
    c = (float*)malloc(n*sizeof(*c));

    fill(A, b, n);
    const double tstart = omp_get_wtime();
    tri_gemv(A, b, c, n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("Elapsed time (s): %f\n", elapsed);
    check(c, n);
    free(A);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}
