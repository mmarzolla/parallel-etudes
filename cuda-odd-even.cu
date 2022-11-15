/****************************************************************************
 *
 * cuda-odd-even.cu - Odd-even sort with CUDA
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
% HPC - Odd-even transposition sort
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-11-18

A lezione è stato discusso l'algoritmo di ordinamento _Odd-Even
Transposition Sort_. L'algoritmo è una variante di BubbleSort, ed è in
grado di ordinare un array di $n$ elementi in tempo $O(n^2)$. Pur non
essendo efficiente, l'algoritmo si presta bene ad essere
parallelizzato. Abbiamo già visto versioni parallele basate su OpenMP
e MPI; in questo esercizio viene richiesta la realizzazione di una
versione CUDA.

Dato un array `v[]` di $n$ elementi, l'algoritmo esegue $n$ fasi
numerate da 0 a $n–1$; nelle fasi pari si confrontano gli elementi di
`v[]` di indice pari con i successivi, scambiandoli se non sono
nell'ordine corretto. Nelle fasi dispari si esegue la stessa
operazione confrontando gli elementi di `v[]` di indice dispari con i
successivi (Figura 1).

![Figura 1: Odd-Even Sort](cuda-odd-even.png)

Il file [cuda-odd-even.cu](cuda-odd-even.cu) contiene una
implementazione dell'algoritmo Odd-Even Transposition
Sort. L'implementazione fornita fa solo uso della CPU: scopo di questo
esercizio è di sfruttare il parallelismo CUDA.

Il paradigma CUDA suggerisce di adottare un parallelismo a grana fine,
facendo gestire ad ogni thread il confronto e lo scambio di una coppia
di elementi adiacenti. La soluzione più semplice consiste nel creare
$n$ CUDA thread e lanciare $n$ volte un kernel che sulla base
dell'indice della fase (da passare come parametro al kernel), attiva i
thread che agiscono sugli elementi dell'array di indice pari o
dispari. In altre parole, la struttura di questo kernel è simile a:

```C
__global__ odd_even_step_bad( int *x, int n, int phase )
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( (idx < n-1) && ((idx % 2) == (phase % 2)) ) {
		cmp_and_swap(&x[idx], &x[idx+1]);
	}
}
```

Questa soluzione non è però efficiente, perché solo metà dei thread
sono attivi durante ogni fase. Realizzare quindi una seconda versione
in cui in ogni fase si lanciano $\lceil n/2 \rceil$ CUDA thread,
facendo in modo che tutti i thread siano sempre attivi durante ogni
fase. Nelle fasi pari i thread $0, 1, 2, 3, \ldots$ gestiranno
rispettivamente le coppie di indici $(0, 1)$, $(2, 3)$, $(4, 5)$, $(6,
7)$, $\ldots$, mentre nelle fasi dispari gestiranno le coppie di
indici $(1, 2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, $\ldots$.

La Tabella 1 illustra la corrispondenza tra l'indice "lineare" dei
thread `idx`, calcolato come nel frammento di codice sopra, e la
coppia di indici dell'array che devono gestire.

:Tabella 1: corrispondenza tra indice dei thread e dell'array

Indice thread      Fasi pari     Fasi dispari
-----------------  ------------  --------------
0                  $(0,1)$       $(1,2)$
1                  $(2,3)$       $(3,4)$
2                  $(4,5)$       $(5,6)$
3                  $(6,7)$       $(7,8)$ 
4                  $(8,9)$       $(9,10)$
...                ...           ... 
-----------------  ------------  --------------

Per compilare:

        nvcc cuda-odd-even.cu -o cuda-odd-even

Per eseguire:

        ./cuda-odd-even [len]

Esempio:

        ./cuda-odd-even 1024

## File

- [cuda-odd-even.cu](cuda-odd-even.cu)
- [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* if *a > *b, swap them. Otherwise do nothing */
#ifndef SERIAL
__host__ __device__
#endif
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

#ifndef SERIAL
#define BLKDIM 1024

/**
 * This kernel requires `n` threads to sort `n` elements, but only
 * half the threads are used during each phase. Therefore, this kernel
 * is not efficient.
 */
__global__ void odd_even_step_bad( int *x, int n, int phase )
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( (idx < n-1) && ((idx % 2) == (phase % 2)) ) {
        /* Compare & swap x[idx] and x[idx+1] */
        cmp_and_swap(&x[idx], &x[idx+1]);
    }
}
#endif

/* Odd-even transposition sort */
void odd_even_sort( int* v, int n )
{
#ifdef SERIAL
    for (int phase = 0; phase < n; phase++) {
        if ( phase % 2 == 0 ) {
            /* (even, odd) comparisons */
            for (int i=0; i<n-1; i += 2 ) {
                cmp_and_swap( &v[i], &v[i+1] );
            }
        } else {
            /* (odd, even) comparisons */
            for (int i=1; i<n-1; i += 2 ) {
                cmp_and_swap( &v[i], &v[i+1] );
            }
        }
    }
#else
    int *d_v; /* device copy of `v` */
    const int NBLOCKS = (n + BLKDIM-1)/BLKDIM;
    const size_t SIZE = n * sizeof(*d_v);

    /* Allocate `d_v` on device */
    cudaSafeCall( cudaMalloc((void **)&d_v, SIZE) );

    /* Copy `v` to device memory */
    cudaSafeCall( cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice) );

    printf("BAD version (%d elements, %d CUDA threads):\n", n, NBLOCKS * BLKDIM);
    for (int phase = 0; phase < n; phase++) {
        odd_even_step_bad<<<NBLOCKS, BLKDIM>>>(d_v, n, phase);
        cudaCheckError();
    }

    /* Copy result back to host */
    cudaSafeCall( cudaMemcpy(v, d_v, SIZE, cudaMemcpyDeviceToHost) );

    /* Free memory on the device */
    cudaFree(d_v);
#endif
}

#ifndef SERIAL
/**
 * A more efficient kernel that uses n/2 threads to sort n elements.
 */
__global__ void odd_even_step_good( int *x, int n, int phase )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x; /* thread index */
    const int idx = tid*2 + (phase % 2); /* array index handled by this thread */
    if (idx < n-1) {
        cmp_and_swap(&x[idx], &x[idx+1]);
    }
}

/* This function is almost identical to odd_even_sort(), with the
   difference that it uses a more efficient kernel
   (odd_even_step_good()) that only requires n/2 threads during each
   phase. */
void odd_even_sort_good(int *v, int n)
{
    int *d_v; /* device copy of v */
    const int NBLOCKS = (n/2 + BLKDIM-1)/BLKDIM;
    const size_t SIZE = n * sizeof(*d_v);

    /* Allocate d_v on device */
    cudaSafeCall( cudaMalloc((void **)&d_v, SIZE) );

    /* Copy v to device memory */
    cudaSafeCall( cudaMemcpy(d_v, v, SIZE, cudaMemcpyHostToDevice) );

    printf("GOOD version (%d elements, %d CUDA threads):\n", n, NBLOCKS * BLKDIM);
    for (int phase = 0; phase < n; phase++) {
        odd_even_step_good<<<NBLOCKS, BLKDIM>>>(d_v, n, phase);
        cudaCheckError();
    }

    /* Copy result back to host */
    cudaSafeCall( cudaMemcpy(v, d_v, SIZE, cudaMemcpyDeviceToHost) );

    /* Free memory on the device */
    cudaFree(d_v);
}
#endif

/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b-a+1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
    for(i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        const int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/**
 * Check correctness of the result
 */
int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != i) {
            fprintf(stderr, "Check FAILED: x[%d]=%d, expected %d\n", i, x[i], i);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] )
{
    int *x;
    int n = 128*1024;
    const int MAX_N = 512*1024*1024;
    double tstart, elapsed;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*x);

    /* Allocate space for x on host */
    x = (int*)malloc(SIZE); assert(x != NULL);
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    /* Check result */
    check(x, n);

#ifndef SERIAL
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort_good(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    check(x, n);
#endif

    /* Cleanup */
    free(x);

    return EXIT_SUCCESS;
}
