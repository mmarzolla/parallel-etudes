/****************************************************************************
 *
 * simd-matmul.c - Dense matrix-matrix multiply using vector datatypes
 *
 * Copyright (C) 2017--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Prodotto matrice-matrice SIMD
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-05-15

Il file [simd-matmul.c](simd-matmul.c) contiene la versione seriale
del prodotto matrice-matrice $r=p \times q$, sia nella versione
normale che nella versione _cache-efficient_ che traspone la matrice
$q$ per sfruttare al meglio la cache (ne abbiamo parlato a lezione).

La versione cache-efficient può essere usata per trarre vantaggio
dalle estensioni SIMD del processore, in quanto il prodotto
riga-colonna diventa un prodotto riga-riga. Osserviamo che il corpo
della funzione `scalar_matmul_tr()`

```C
for (i=0; i<n; i++) {
	for (j=0; j<n; j++) {
		double s = 0.0;
		for (k=0; k<n; k++) {
			s += p[i*n + k] * qT[j*n + k];
		}
		r[i*n + j] = s;
	}
}
```

calcola il prodotto scalare di due vettori di $n$ elementi memorizzati
a partire dagli indirizzi di memoria $(p + i \times n)$ e
$(\mathit{qT} + j \times n)$. Sfruttando il calcolo SIMD del prodotto
scalare, realizzare la funzione `simd_matmul_tr()` in cui il prodotto
scalare di cui sopra viene calcolato usando i _vector datatype_ del
compilatore. Si garantisce che le dimensioni delle matrici siano
multiple della lunghezza dei vettori SIMD.

Si presti attenzione che in questo esercizio si usa il tipo `double`;
è pertanto necessario definire un tipo vettoriale `v2d`, di ampiezza
16 Byte e composto da due valori `double`, utilizzando una
dichiarazione simile a quella vista a lezione:

```C
typedef double v2d __attribute__((vector_size(16)));
#define VLEN (sizeof(v2d)/sizeof(double))
```

Il server usato per le esercitazioni supporta le estensioni SIMD di
Intel fino ad AVX2, e quindi dispone di registri SIMD di ampiezza 256
bit = 32 Byte. Si provi a modificare la propria implementazione per
sfruttare un tipo vettoriale `v4d` contenente quattro valori `double`.

Compilare con:

        gcc -march=native -O2 -std=c99 -Wall -Wpedantic -D_XOPEN_SOURCE=600 simd-matmul.c -o simd-matmul

Eseguire con

        ./simd-matmul [matrix size]

Esempio:

        ./simd-matmul 1024

## File

- [simd-matmul.c](simd-matmul.c)
- [hpc.h](hpc.h)

***/

/* The following #define is required by posix_memalign() */
#define _XOPEN_SOURCE 600

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>  /* for assert() */
#include <strings.h> /* for bzero() */

typedef double v2d __attribute__((vector_size(16)));
#define VLEN (sizeof(v2d)/sizeof(double))

/* Fills n x n square matrix m */
void fill( double* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            m[i*n + j] = (i%10+j) / 10.0;
        }
    }
}

/* compute r = p * q, where p, q, r are n x n matrices. */
void scalar_matmul( const double *p, const double* q, double *r, int n)
{
    int i, j, k;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            double s = 0.0;
            for (k=0; k<n; k++) {
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
    int i, j, k;
    double *qT = (double*)malloc( n * n * sizeof(*qT) );

    assert(qT != NULL);

    /* transpose q, storing the result in qT */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            double s = 0.0;
            for (k=0; k<n; k++) {
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
    /* [TODO] Implement this function */
#else
    int i, j, k;
    double *qT;
    const v2d *vp, *vqT;
    int ret;

    ret = posix_memalign((void**)&qT, __BIGGEST_ALIGNMENT__, n*n*sizeof(*qT));
    assert( 0 == ret );

    /* transpose q, storing the result in qT */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }

    /* multiply p and qT row-wise using SIMD intrinsics */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            v2d vs = {0.0, 0.0};
            vp  = (v2d*)(p + i*n);
            vqT = (v2d*)(qT + j*n);
            for (k=0; k<n-VLEN+1; k += VLEN) {
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
