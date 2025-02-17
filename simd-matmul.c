/****************************************************************************
 *
 * simd-matmul.c - Dense matrix-matrix product using vector datatypes
 *
 * Copyright (C) 2017--2023 Moreno Marzolla
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
% HPC - Dense matrix-matric product using vector datatypes
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2023-03-07

The file [simd-matmul.c](simd-matmul.c) contains a serial version of
the dense matrix-matrix product of two square matrices $p, q$, $r=p
\times q$. Both a "plain" and cache-efficient version are provided;
the cache-efficient program transposes $q$ so that the product can be
computed by accessing both $p$ and $q^T$ row-wise.

The cache-efficient matrix-matrix product can be modified to take
advantage of SIMD instructions, since it essentially computes a number
of dot products between rows of $p$ and rows of $q^T$. Indeed, the
body of function `scalar_matmul_tr()`

```C
for (int i=0; i<n; i++) {
	for (int j=0; j<n; j++) {
		double s = 0.0;
		for (int k=0; k<n; k++) {
			s += p[i*n + k] * qT[j*n + k];
		}
		r[i*n + j] = s;
	}
}
```

computes the dot product of two arrays of length $n$ that are stored
at memory addresses $(p + i \times n)$ and $(\mathit{qT} + j \times
n)$, respectively.

Your goal is to use SIMD instructions to compute the dot product above
using _vector datatypes_ provided by the GCC compiler. The program
guarantees that the array length $n$ is an integer multiple of the
SIMD register length, and that all data are suitably aligned in
memory.

This exercise uses the `double` data type; it is therefore necessary
to define a vector datatype `v2d` of length 16 Bytes containing two
doubles, using the declaration:

```C
	typedef double v2d __attribute__((vector_size(16)));
	#define VLEN (sizeof(v2d)/sizeof(double))
```

The server `isi-raptor03` supports the AVX2 instruction set, and
therefore has SIMD registers of width 256 bits = 32 Bytes. You might
want to make use of a wider datatype `v4d` containing 4 doubles
instead of two.

To compile:

        gcc -march=native -O2 -std=c99 -Wall -Wpedantic simd-matmul.c -o simd-matmul

To execute:

        ./simd-matmul [matrix size]

Example:

        ./simd-matmul 1024

## Files

- [simd-matmul.c](simd-matmul.c)
- [hpc.h](hpc.h)

***/

/* The following #define is required by posix_memalign() */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>  /* for assert() */
#include <strings.h> /* for bzero() */

#include "hpc.h"

/* This program works on double-precision numbers; therefore, we
   define a v2d vector datatype that contains two doubles in a SIMD
   array of 16 bytes (VLEN==2). */
typedef double v2d __attribute__((vector_size(16)));
#define VLEN (sizeof(v2d)/sizeof(double))

/* Fills n x n square matrix m */
void fill( double* m, int n )
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            m[i*n + j] = (i%10+j) / 10.0;
        }
    }
}

/* compute r = p * q, where p, q, r are n x n matrices. */
void scalar_matmul( const double *p, const double* q, double *r, int n)
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<n; k++) {
                s += p[i*n + k] * q[k*n + j];
            }
            r[i*n + j] = s;
        }
    }
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
   matrices. This function allocates (and then releases) an additional n x n
   temporary matrix. */
void scalar_matmul_tr( const double *p, const double* q, double *r, int n)
{
    double *qT = (double*)malloc( n * n * sizeof(*qT) );

    assert(qT != NULL);

    /* transpose q, storing the result in qT */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<n; k++) {
                s += p[i*n + k] * qT[j*n + k];
            }
            r[i*n + j] = s;
        }
    }

    free(qT);
}

/* SIMD version of the cache-efficient matrix-matrix multiply above.
   This function requires that n is a multiple of the SIMD vector
   length VLEN */
void simd_matmul_tr( const double *p, const double* q, double *r, int n)
{
#ifdef SERIAL
    assert(n % VLEN == 0);
    /* [TODO] Implement this function */
#else
    double *qT;
    int ret = posix_memalign((void**)&qT, __BIGGEST_ALIGNMENT__, n*n*sizeof(*qT));
    assert( 0 == ret );
    assert(n % VLEN == 0);

    /* transpose q, storing the result in qT */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise using vector datatypes */
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            v2d vs = {0.0, 0.0};
            const v2d *vp  = (v2d*)(p + i*n);
            const v2d *vqT = (v2d*)(qT + j*n);
            for (int k=0; k<n-VLEN+1; k += VLEN) {
                vs += (*vp) * (*vqT);
                vp++;
                vqT++;
            }
            r[i*n + j] = vs[0] + vs[1];
        }
    }

    free(qT);
#endif
}

int main( int argc, char* argv[] )
{
    int n = 512;
    double *p, *q, *r;
    double tstart, elapsed, tserial;
    int ret;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 != n % VLEN ) {
        fprintf(stderr, "FATAL: the matrix size must be a multiple of %d\n", (int)VLEN);
        return EXIT_FAILURE;
    }

    const size_t size = n*n*sizeof(*p);

    ret = posix_memalign((void**)&p, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&q, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );
    ret = posix_memalign((void**)&r, __BIGGEST_ALIGNMENT__,  size);
    assert( 0 == ret );

    fill(p, n);
    fill(q, n);
    printf("\nMatrix size: %d x %d\n\n", n, n);

    tstart = hpc_gettime();
    scalar_matmul(p, q, r, n);
    tserial = elapsed = hpc_gettime() - tstart;
    printf("Scalar\t\tr[0][0] = %f, Exec time = %f\n", r[0], elapsed);

    bzero(r, size);

    tstart = hpc_gettime();
    scalar_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("Transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial/elapsed );

    bzero(r, size);

    tstart = hpc_gettime();
    simd_matmul_tr(p, q, r, n);
    elapsed = hpc_gettime() - tstart;
    printf("SIMD transposed\tr[0][0] = %f, Exec time = %f (speedup vs scalar %.2fx)\n", r[0], elapsed, tserial/elapsed);

    free(p);
    free(q);
    free(r);
    return EXIT_SUCCESS;
}
