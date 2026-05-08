/****************************************************************************
 *
 * omp-tri-gemv.c - Upper-triangular Matrix-Vector multiply
 *
 * Copyright (C) 2024, 2026 Moreno Marzolla
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
% Last updated: 2026-05-08

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

/* First solution: parallelize the outer loop. Use `schedule(dynamic)`
   to reduce load unbalance. */
void tri_gemv1(const float *A, const float *b, float *c, int n)
{
#ifndef SERIAL
    /* arbitrary cunksize. */
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int i=0; i<n; i++) {
        c[i] = 0;
        for (int j=i; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}

/* Second solution: parallelize the inner loop. Might want to use the
   `if()` clause to avoid creating a team of threads if theer is too
   little work to do. */
void tri_gemv2(const float *A, const float *b, float *c, int n)
{
    for (int i=0; i<n; i++) {
        c[i] = 0;
#pragma omp parallel for reduction(+:c[i])
        for (int j=i; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}

/* Third solution: parallelize both loops. Requires OpenMP with
   support of non-rectangular loop nests; also, requires
   initialization of `c[]` to be moved outside the loop.
   Note the use of array reduction. */
void tri_gemv3(const float *A, const float *b, float *c, int n)
{
#pragma opm parallel
    {
#pragma omp for
        for (int i=0; i<n; i++)
            c[i] = 0.0f;
#pragma omp for collapse(2) reduction(+:c[:n])
        for (int i=0; i<n; i++) {
            for (int j=i; j<n; j++) {
                c[i] += A[i*n + j] * b[j];
            }
        }
    }
}

/* Fourth solution: parallelize the outer loop, assign A[i][] and
   A[n-1-i][] to the same thread to reduce load unbalance. */
void tri_gemv4(const float *A, const float *b, float *c, int n)
{
#pragma omp parallel for
    for (int i=0; i<(n+1)/2; i++) {
        /* row i. */
        const int iu = i;
        c[iu] = 0;
        for (int j=iu; j<n; j++) {
            c[iu] += A[iu*n + j] * b[j];
        }

        /* Row (n-1-i). */
        const int id = n-1-i;
        if (id != iu) {
            c[id] = 0;
            for (int j=id; j<n; j++) {
                c[id] += A[id*n + j] * b[j];
            }
        }
    }
}

typedef void(*tri_gemv_t)(const float *, const float *, float *, int);

void check(const char *name, tri_gemv_t f, const float *A, const float *b, int n)
{
    static const float EPS = 1e-5;
    float *c = (float*)malloc(n*sizeof(*c));
    /* Fill `c` with wrong values. */
    for (int i=0; i<n; i++)
        c[i] = i;
    printf("%s\t", name);
    const double tstart = omp_get_wtime();
    f(A, b, c, n);
    const double elapsed = omp_get_wtime() - tstart;
    int test_ok = 1;
    int i;
    for (i=0; i<n && test_ok; i++) {
        const float expected = n-i;
        test_ok = fabs(c[i] - expected) <= EPS;
    }
    if (test_ok)
        printf("OK\t");
    else
        printf("FAILED: c[%d]=%f, expected %f\t", i, c[i], (float)(n-i));
    printf("Execution time %.3f\n", elapsed);
    free(c);
}

int main( int argc, char *argv[] )
{
    const int MAXN = 20000;
    int n = 100;
    float *A, *b;

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

    fill(A, b, n);
    check("Parallel outer loop", tri_gemv1, A, b, n);
    fill(A, b, n); /* invalidate the cache. */
    check("Parallel inner loop", tri_gemv2, A, b, n);
    fill(A, b, n); /* ditto */
    check("Both loops collapsed", tri_gemv3, A, b, n);
    fill(A, b, n); /* ditto */
    check("Symmetric outer loop", tri_gemv4, A, b, n);

    free(A);
    free(b);
    return EXIT_SUCCESS;
}
