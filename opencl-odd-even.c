/****************************************************************************
 *
 * opencl-odd-even.c - Odd-even sort
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
% Ultimo aggiornamento: 2021-12-04

A lezione è stato discusso l'algoritmo di ordinamento _Odd-Even
Transposition Sort_. L'algoritmo è una variante di BubbleSort, e
ordina un array di $n$ elementi in tempo $O(n^2)$. Pur non essendo
efficiente, l'algoritmo si presta bene ad essere
parallelizzato. Abbiamo già visto versioni parallele basate su OpenMP
e MPI; in questo esercizio viene richiesta la realizzazione di una
versione OpenCL.

Dato un array `v[]` di $n$ elementi, l'algoritmo esegue $n$ fasi
numerate da 0 a $n–1$; nelle fasi pari si confrontano gli elementi di
`v[]` di indice pari con i successivi, scambiandoli se non sono
nell'ordine corretto. Nelle fasi dispari si esegue la stessa
operazione confrontando gli elementi di `v[]` di indice dispari con i
successivi (Figura 1).

![Figura 1: Odd-Even Sort](opencl-odd-even.png)

Il file [opencl-odd-even.c](opencl-odd-even.c) contiene una
implementazione dell'algoritmo Odd-Even Transposition
Sort. L'implementazione fornita fa solo uso della CPU: scopo di questo
esercizio è di sfruttare il parallelismo OpenCL.

Il paradigma OpenCL suggerisce di adottare un parallelismo a grana
fine, facendo gestire ad ogni work-item il confronto e lo scambio di
una coppia di elementi adiacenti. La soluzione più semplice consiste
nel creare $n$ work-item e lanciare $n$ volte un kernel che sulla base
dell'indice della fase (da passare come parametro al kernel), attiva i
work-item che agiscono sugli elementi dell'array di indice pari o
dispari. In altre parole, la struttura di questo kernel è simile a:

```C
__kernel step_bad( __global int *x, int n, int phase )
{
	const int idx = get_global_id(0);
	if ( (idx < n-1) && ((idx % 2) == (phase % 2)) ) {
		cmp_and_swap(&x[idx], &x[idx+1]);
	}
}
```

Questa soluzione non è però efficiente, perché solo metà dei work-item
sono attivi durante ogni fase. Realizzare quindi una seconda versione
in cui in ogni fase si lanciano $\lceil n/2 \rceil$ work-item, facendo
in modo che tutti siano sempre attivi durante ogni fase. Nelle fasi
pari i work-item con id globale $0, 1, 2, 3, \ldots$ gestiranno
rispettivamente le coppie di indici $(0, 1)$, $(2, 3)$, $(4, 5)$, $(6,
7)$, $\ldots$, mentre nelle fasi dispari gestiranno le coppie di
indici $(1, 2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, $\ldots$.

La Tabella 1 illustra la corrispondenza tra l'indice globale `idx` dei
work-item e la coppia di indici dell'array che devono gestire.

:Tabella 1: corrispondenza tra indice dei work-item e dell'array

Indice globale     Fasi pari     Fasi dispari
-----------------  ------------  --------------
0                  $(0,1)$       $(1,2)$
1                  $(2,3)$       $(3,4)$
2                  $(4,5)$       $(5,6)$
3                  $(6,7)$       $(7,8)$
4                  $(8,9)$       $(9,10)$
...                ...           ...
-----------------  ------------  --------------

> **Attenzione.** L'hardware impone delle limitazioni sulla dimensione
> della coda dei comandi OpenCL. In altre parole, un frammento di
> codice come questo:
>
> ```C
> for (int phase = 0; phase < n; phase++) {
>    sclSetArgsEnqueueKernel(...);
> }
> ```
>
> potrebbe causare dei crash (es., _segmentation fault_) o altri
> errori, soprattutto per valori elevati di _n_. La libreria
> `simpleCL` prende precauzioni, ed inserisce periodicamente il
> comando `sclDeviceSynchronize()` dopo l'inserimento di un kernel in
> coda.

Per compilare:

        cc opencl-odd-even.c simpleCL.c -o opencl-odd-even -lOpenCL

Per eseguire:

        ./opencl-odd-even [len]

Esempio:

        ./opencl-odd-even 1024

## File

- [opencl-odd-even.c](opencl-odd-even.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h)
- [hpc.h](hpc.h)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

#ifndef SERIAL
sclKernel step_kernel_bad, step_kernel_good;
#endif

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

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
    cl_mem d_v; /* device copy of `v` */
    const size_t SIZE = n * sizeof(*v);
    const size_t GLOBAL_SIZE = sclRoundUp(n, SCL_DEFAULT_WG_SIZE);

    d_v = sclMallocCopy(SIZE, v, CL_MEM_READ_WRITE);

    printf("BAD version (%d elements, %d work-items):\n", n, (int)GLOBAL_SIZE);
    for (int phase = 0; phase < n; phase++) {
        sclSetArgsEnqueueKernel(step_kernel_bad,
                                DIM1(GLOBAL_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                                ":b :d :d",
                                d_v, n, phase);
    }

    /* Copy result back to host */
    sclMemcpyDeviceToHost(v, d_v, SIZE);

    /* Free memory on the device */
    sclFree(d_v);
#endif
}

#ifndef SERIAL
/* This function is almost identical to odd_even_sort(), with the
   difference that it uses a more efficient kernel
   (odd_even_step_good()) that only requires n/2 work-items during
   each phase. */
void odd_even_sort_good(int *v, int n)
{
    cl_mem d_v; /* device copy of v */
    const size_t GLOBAL_SIZE = sclRoundUp(n/2, SCL_DEFAULT_WG_SIZE);
    const size_t SIZE = n * sizeof(*v);

    d_v = sclMallocCopy(SIZE, v, CL_MEM_READ_WRITE);

    printf("GOOD version (%d elements, %d work-items):\n", n, (int)GLOBAL_SIZE);
    for (int phase = 0; phase < n; phase++) {
        sclSetArgsEnqueueKernel(step_kernel_good,
                                DIM1(GLOBAL_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                                ":b :d :d",
                                d_v, n, phase);
    }

    /* Copy result back to host */
    sclMemcpyDeviceToHost(v, d_v, SIZE);

    /* Free memory on the device */
    sclFree(d_v);
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

#ifndef SERIAL
    sclInitFromFile("opencl-odd-even.cl");
    step_kernel_bad = sclCreateKernel("step_kernel_bad");
    step_kernel_good = sclCreateKernel("step_kernel_good");
#endif
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

#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
