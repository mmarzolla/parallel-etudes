/****************************************************************************
 *
 * omp-tri-gemv.c - Upper-triangular Matrix-Vector multiply
 *
 * Copyright (C) <YEAR> Moreno Marzolla
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
% HPC - Upper-triangular Matrix-Vector multiply
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-12-03

***/

#include <stdio.h>
#include <stdlib.h>

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
#pragma omp parallel
    for (int i=0; i<n; i++) {
#pragma omp single
        c[i] = 0;
#pragma omp for reduction(+:c[i])
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

int main( void )
{
    const int N = 100;
    float *A = (float*)malloc(N*N*sizeof(*A));
    float *b = (float*)malloc(N*sizeof(*b));
    float *c = (float*)malloc(N*sizeof(*c));

    fill(A, b, N);
    tri_gemv(A, b, c, N);
    check(c, N);
    free(A);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}
