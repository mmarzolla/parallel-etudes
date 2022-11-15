/****************************************************************************
 *
 * omp-schedule.c - simulate "schedule(static)" and "schedule(dynamic)" using the "omp parallel" clause
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
% HPC - Realizzazione delle clausole "schedule(static)" e "schedule(dynamic)"
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-10-24

A lezione abbiamo visto come sia possibile utilizzare le clausole
`schedule(static)` e `schedule(dynamic)` per assegnare le iterazioni
di un ciclo "for" ai thread OpenMP. Lo scopo di questo esercizio è di
simulare queste clausole _senza_ usare il costrutto `omp parallel
for`.

Il file [omp-schedule.c](omp-schedule.c) contiene una implementazione
seriale di un programma che effettua le seguenti computazioni. Il
programma crea e inizializza un array `vin[]` di $n$ interi (il valore
$n$ può essere passato da riga di comando). Il programma crea un
secondo array `vout[]`, sempre di lunghezza $n$, e ne definisce il
contenuto in modo tale che `vout[i] = Fib(vin[i])` per ogni $i$,
essendo `Fib(k)` il _k_-esimo numero della successione di Fibonacci:
`Fib(0)` = `Fib(1)` = 1; `Fib(k) = Fib(k-1) + Fib(k-2)` se $k \geq
2$. Il calcolo dei numeri di Fibonacci viene fatto in modo volutamente
inefficiente con un algoritmo ricorsivo, per far sì che il tempo di
calcolo vari significativamente al variare di $i$.

Sono presenti due funzioni identiche, `do_static()` e `do_dynamic()`,
contenenti un ciclo "for" che realizza il calcolo di cui sopra.

1. Modificare la funzione `do_static()` per distribuire le iterazioni
   del ciclo come se fosse presente la direttiva `schedule(static,
   chunk_size)`, ma senza usare il costrutto `omp parallel parallel`
   (è invece consentito usare `omp parallel`).

2. Fatto ciò, si modifichi la funzione `do_dynamic()` per distribuire
   le iterazioni del ciclo come se fosse presente una direttiva
   `schedule(dynamic, chunk_size)`. Anche in questo caso non si deve
   usare la direttiva `omp parallel for`.

**Suggerimenti.** In entrambi i casi consiglio di iniziare assumendo
`chunk_size = 1`. Per il punto 2 (schedule dinamico) consiglio di
procedere come segue: supponendo `chunk_size = 1`, si utilizza una
variabile condivisa per indicare qual è l'indice del prossimo elemento
di `vin[]` che deve essere assegnato ad un thread. Ogni thread
acquisisce in mutua esclusione il prossimo elemento di `vin[]`, se
presente, e lo elabora in maniera indipendente dagli altri thread.

Se avanza tempo, si confrontino i tempi di esecuzione della propria
implementazione con quelli ottenuti applicando le clausole
`schedule(static, chunk_size)` e `schedule(dynamic, chunk_size)` al
ciclo "for".

Per compilare:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-schedule.c -o omp-schedule

Per eseguire:

        ./omp-schedule [n]

Esempio:

        OMP_NUM_THREADS=2 ./omp-schedule

## File

- [omp-schedule.c](omp-schedule.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Recursive computation of the n-th Fibonacci number, for n=0, 1, 2, ...
   Do not parallelize this function. */
int fib_rec(int n)
{
    if (n<2) {
        return 1;
    } else {
        return fib_rec(n-1) + fib_rec(n-2);
    }
}

/* Iterative computation of the n-th Fibonacci number. This function
   must be used for checking the result only. */
int fib_iter(int n)
{
    if (n<2) {
        return 1;
    } else {
        int fibnm1 = 1;
        int fibnm2 = 1;
        int fibn;
        n = n-1;
        do {
            fibn = fibnm1 + fibnm2;
            fibnm2 = fibnm1;
            fibnm1 = fibn;
            n--;
        } while (n>0);
        return fibn;
    }
}

/* Fill vectors `vin` and `vout` of length `n`; `vin` will contain
   input values; `vout` is initialized with -1 */
void fill(int *vin, int *vout, int n)
{
    int i;
    /* fill input array */
    for (i=0; i<n; i++) {
        vin[i] = 25 + (i%10);
        vout[i] = -1;
    }
}

/* Check correctness of `vout[]`. Return 1 if correct, 0 if not */
int check(const int *vin, const int *vout, int n)
{
    int i;
    /* check result */
    for (i=0; i<n; i++) {
        if ( vout[i] != fib_iter(vin[i]) ) {
            fprintf(stderr,
                    "Test FAILED: vin[%d]=%d, vout[%d]=%d (expected %d)\n",
                    i, vin[i], i, vout[i], fib_iter(vin[i]));
            return 0;
        }
    }
    fprintf(stderr, "Test OK\n");
    return 1;
}


void do_static(const int *vin, int *vout, int n)
{
    int i;
#ifdef SERIAL
    /* [TODO] parallelize the following loop, simulating a
       "schedule(static,1)" clause, i.e., static scheduling with block
       size 1. Do not modify the body of the fib_rec() function. */
    for (i=0; i<n; i++) {
        vout[i] = fib_rec(vin[i]);
        /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
    }
#else
    const int chunk_size = 1; /* can be set to any value >= 1 */
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(vin,vout,n) private(i)
#else
#pragma omp parallel default(none) shared(vin,vout,n,chunk_size) private(i)
#endif
    {
        /* This implementation simulates the behavior of a
           schedule(static,chunk_size) clause for any chunk_size>=1. */
        const int my_id = omp_get_thread_num();
        const int n_threads = omp_get_num_threads();
        const int START = my_id * chunk_size;
        const int STRIDE = n_threads * chunk_size;
        int j;

        for (i=START; i<n; i += STRIDE) {
            for (j=i; j<i+chunk_size && j<n; j++) {
                vout[j] = fib_rec(vin[j]);
                /* printf("Thread %d vin[%d]=%d vout[%d]=%d\n", tid, j, vin[j], j, vout[j]); */
            }
        }
    }
#endif
}

void do_dynamic(const int *vin, int *vout, int n)
{
    int i;
#ifdef SERIAL
    /* [TODO] parallelize the following loop, simulating a
       "schedule(dynamic,1)" clause, i.e., dynamic scheduling with
       block size 1. Do not modify the body of the fib_rec()
       function. */
    for (i=0; i<n; i++) {
        vout[i] = fib_rec(vin[i]);
        /* printf("vin[%d]=%d vout[%d]=%d\n", i, vin[i], i, vout[i]); */
    }
#else
    int idx = 0; /* shared index */
    const int chunk_size = 1; /* can be set to any value >= 1 */
#if __GNUC__ < 9
#pragma omp parallel default(none) shared(idx,vin,vout,n) private(i)
#else
#pragma omp parallel default(none) shared(idx,vin,vout,n,chunk_size) private(i)
#endif
    {
        /* This implementation simulates the behavior of a
           schedule(dynamic,chunk_size) clause for any chunk_size>=1. */
        int my_idx;
        do {
            /* atomically grab current index, and increment */
#pragma omp critical
            {
                my_idx = idx;
                idx += chunk_size;
            }
            for (i=my_idx; i<my_idx+chunk_size && i<n; i++) {
                vout[i] = fib_rec(vin[i]);;
                /* printf("Thread %d vin[%d]=%d vout[%d]=%d\n", tid, i, vin[j], j, vout[j]); */
            }
        } while (my_idx < n);
    }
#endif
}

int main( int argc, char* argv[] )
{
    int n = 1024;
    const int max_n = 512*1024*1024;
    int *vin, *vout;
    double tstart, elapsed;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > max_n ) {
        fprintf(stderr, "FATAL: n too large\n");
        return EXIT_FAILURE;
    }

    /* initialize the input and output arrays */
    vin = (int*)malloc(n * sizeof(vin[0])); assert(vin != NULL);
    vout = (int*)malloc(n * sizeof(vout[0])); assert(vout != NULL);

    /**
     ** First test
     **/
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_static(vin, vout, n);
    elapsed = omp_get_wtime() - tstart;
    check(vin, vout, n);

    printf("Elapsed time (static schedule): %f\n", elapsed);

    /**
     ** First test
     **/
    fill(vin, vout, n);
    tstart = omp_get_wtime();
    do_dynamic(vin, vout, n);
    elapsed = omp_get_wtime() - tstart;
    check(vin, vout, n);

    printf("Elapsed time (dynamic schedule): %f\n", elapsed);

    free(vin);
    free(vout);
    return EXIT_SUCCESS;
}
