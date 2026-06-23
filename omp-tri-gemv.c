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
% Last updated: 2026-06-29

The function `tri_gemv(A, b, c)` computes $c = Ab$, where $A$ is a $n
\times n$ upper triangular matrix and $b$ a vector. The goal of this
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

#ifdef SERIAL
void tri_gemv(const float *A, const float *b, float *c, int n)
{
    for (int i=0; i<n; i++) {
        c[i] = 0;
        for (int j=i; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}
#else
/* First solution: parallelize the outer loop. Use `schedule(dynamic)`
   to reduce load unbalance. */
void tri_gemv1(const float *A, const float *b, float *c, int n)
{
    /* Chunksize is set arbitrarily. */
#pragma omp parallel for schedule(dynamic, 64)
    for (int i=0; i<n; i++) {
        c[i] = 0;
        for (int j=i; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
}

/* Second solution: parallelize the inner loop. Might want to use the
   `if()` clause to avoid creating a parallel region if there is too
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

/* Third solution: parallelize both loops. Requires support of
   non-rectangular loop nests (OpenMP >= 5.1). Initialization of `c[]`
   must be moved outside the loop. Note the use of array reduction. */
void tri_gemv3(const float *A, const float *b, float *c, int n)
{
#pragma omp parallel
    {
#pragma omp for
        for (int i=0; i<n; i++)
            c[i] = 0.0f;
#if _OPENMP < 202011
        /* OpenMP < 5.1. It is not possible to collapse both loops, so
           we parallelize the outer loop only.  The schedule() clause
           tries to address load imbalancing caused by the fact that
           for each iteration of the outer loop the work decreases.
           The optimal chunksize should be empirically determined. */
#pragma omp for reduction(+:c[:n]) schedule(dynamic,16)
#else
        /* OpenMP >= 5.1; collapse both loops, to address load
           imbalance. */
#pragma omp for collapse(2) reduction(+:c[:n])
#endif
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

/*

  Out of curiosity, I tried the programs above on the following
  machines:

  1. 12th Gen Intel(R) Core(TM) i9-12900F, 8 P/cores with
  hyperthreading, 8 E/cores, 64GB RAM, Ubuntu 24.04, gcc
  13.3.0-6ubuntu2~24.04.1, using OMP_NUM_THREADS=8;

  2. Intel(R) Core(TM) i5-2500 CPU @ 3.30GHz, 4 cores (no
  hyperthreading), 24GB RAM, Debian 13, gcc Debian 14.2.0-19, using
  all cores.

  compiled with `-fopenmp -Wall -Wpedantic -std=c99 -lm`, matrix size
  20000, times in seconds, average of 5 runs.

  Version                          i9      i5
  ---------------------------- ------  ------
  Outer loop                    0.050   0.186
  Inner loop                    0.177   0.217
  Collapse                      0.055   0.194
  Outer loop, balanced          0.046   0.186

  On both processors, `tri_gemv4()` appears to be the fastest
  version, with `tri_gemv1()` following closely.

 */
#endif

void fill(float *A, float *b, int n)
{
    for (int i=0; i<n; i++) {
        b[i] = 1;
        for (int j=0; j<n; j++) {
            A[i*n + j] = (j >= i);
        }
    }
}

typedef void(*tri_gemv_t)(const float *, const float *, float *, int);

void check(const char *name, tri_gemv_t f, int n)
{
    static const float EPS = 1e-5;
    float *A = (float*)malloc(n*n*sizeof(*A));
    float *b = (float*)malloc(n*sizeof(*b));
    float *c = (float*)malloc(n*sizeof(*c));
    double elapsed = 0;
    const int NREP = 5; /* number of runs. */

    for (int r=0; r<NREP; r++) {
        printf("\r%-25s %d of %d\t", name, r+1, NREP); fflush(stdout);
        fill(A, b, n);
        /* Fill `c` with wrong values. */
        for (int i=0; i<n; i++)
            c[i] = i;
        const double tstart = omp_get_wtime();
        f(A, b, c, n);
        elapsed += omp_get_wtime() - tstart;
    }

    /* Check result of last run. */
    int i, test_ok = 1;
    float expected;
    for (i=0; i<n && test_ok; i++) {
        expected = n-i;
        test_ok = fabs(c[i] - expected) <= EPS;
    }
    if (test_ok)
        printf("Check OK\t");
    else
        printf("Check FAILED: c[%d]=%f, expected %f\t", i, c[i], expected);
    printf("Execution time %.3f\n", elapsed/NREP);

    free(A);
    free(b);
    free(c);
}

int main( int argc, char *argv[] )
{
    const int MAXN = 20000;
    int n = 100;

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

#ifdef SERIAL
    check("Serial", tri_gemv, n);
#else
    check("Parallel outer loop", tri_gemv1, n);
    check("Parallel inner loop", tri_gemv2, n);
    check("Both loops collapsed", tri_gemv3, n);
    check("Symmetric outer loop", tri_gemv4, n);
#endif

    return EXIT_SUCCESS;
}
