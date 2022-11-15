/****************************************************************************
 *
 * cuda-knapsack.c - Solve the 01 integer knapsack problem using CUDA
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
% HPC - Problema dello zaino 0-1
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-05-15

Il _problema dello zaino_ (_knapsack problem_) è un famiglia di
problemi di ottimizzazione combinatoria. In questo esercizio
consideriamo la formulazione seguente, detta problema dello zaino 0/1.
Disponiamo di $n$ oggetti di pesi interi $w_0, \ldots, w_{n-1}$ e
valori reali $v_0, \ldots, v_{n-1}$. Vogliamo determinare il
sottoinsieme di oggetti di peso complessivo minore o uguale a $C$ il
cui valore totale sia massimo possibile.

Questo problema si risolve usando la programmazione dinamica. Per
farlo, consideriamo la famiglia di problemi $P(i,j)$, $i=0, \ldots,
(n-1)$, $j=0, \ldots, C$ definiti nel modo seguente:

> $P(i,j)$ consiste nel determinare il sottoinsieme degli oggetti
> scelti tra $\{0, \ldots, i\}$ aventi peso complessivo minore o
> uguale a $j$ e valore totale massimo possibile.

La soluzione del nostro problema è $P(n-1, C)$.

Sia $V(i,j)$ la soluzione di $P(i,j)$; in altre parole, $V(i,j)$ è il
massimo valore ottenibile da un opportuno sottoinsieme di oggetti
scelti tra $\{0, \ldots, i\}$ il cui peso totale sia minore o uguale a
$j$.

Sappiamo che $V(i,0) = 0$ per ogni $i$ (in un contenitore di capienza
zero non possiamo inserire alcun oggetto, per cui il valore totale
sarà zero).

Che cosa possiamo dire di $V(0,j)$? Avendo a disposizione solo il
primo oggetto, di peso $w_0$ e valore $v_0$, l'unica scelta è di
usarlo oppure no. Lo possiamo inserire nel contenitore se la capienza
$j$ è maggiore o uguale a $w_0$, e in tal caso il valore massimo
ottenibile è $v_0$. Possiamo quindi scrivere:

$$
V(0,j) = \begin{cases}
0 & \mbox{se}\ j < w_0\\
v_0 & \mbox{altrimenti}
\end{cases}
$$

Cosa possiamo dire di $V(i,j)$ nel caso generale? Supponiamo di
conoscere le soluzioni di tutti i problemi "più piccoli", e
concentriamoci sull'oggetto $i$-esimo. Abbiamo due possibilità:

1. Se il peso $w_i$ dell'oggetto $i$ è maggiore della capienza $j$,
   allora sicuramente non possiamo inserirlo nello zaino.  Di
   conseguenza il valore ottimo della soluzione $V(i,j)$ coincide con
   il valore della soluzione $V(i-1, j)$, in cui  ci limitiamo
   a scegliere tra gli oggetti $\{0, 1, \ldots, i-1\}$:

2. Se il peso $w_i$ dell'oggetto $i$ è minore o uguale alla capienza
   $j$, allora potremmo usarlo oppure no.

    a. Se usiamo l'oggetto $i$, il valore massimo degli oggetti nello
       zaino sarà uguale a $v_i$ più il valore massimo che possiamo
       ottenere inserendo un sottoinsieme dei rimanenti $i-1$ oggetti
       nel contenitore, che a questo punto ha capienza residua $j -
       w_i$. Quindi, se usiamo l'oggetto $i$, il valore massimo
       ottenibile è $V(i-1,j-w_i)+v_i$.

    b. Se non usiamo l'oggetto $i$, il valore massimo ottenibile è
       $V(i-1,j)$ (come il caso 1 sopra).

Tra le due opzioni 2.a e 2.b scegliamo quella che produce il valore
massimo. Otteniamo quindi la seguente relazione generale, che vale per
ogni $i=1, \ldots, (n-1)$, $j=0, \ldots, C$:

$$
V(i,j) = \begin{cases}
V(i-1, j) & \mbox{se}\ j < w_i\\
\max\{V(i-1, j), V(i-1, j-w_i) + v_i\} & \mbox{altrimenti}
\end{cases}
$$

In questo esercizio ci limitiamo a calcolare il valore complessivo
degli oggetti appartenenti alla soluzione ottima, piuttosto che la
lista degli oggetti che fanno parte della soluzione.

Il file [cuda-knapsack.cu](cuda-knapsack.cu) risolve il problema dello
zaino usando solo la CPU. Il programma legge una istanza del problema
da un file il cui nome deve essere passato come unico parametro sulla
riga di comando, e visualizza su standard output il valore massimo
degli oggetti che è possibile inserire nello zaino. Il file di input
ha una struttura molto semplice: le prime due righe contengono i
valori di $C$ ed $n$, rispettivamente; seguono $n$ righe ciascuna
delle quali contenente il peso $w_i$ (intero) e il valore $v_i$
(reale) dell'oggetto $i$. Il programma
[knapsack-gen.c](knapsack-gen.c) può essere usato per generare altre
istanze di input.

Modificare il programma fornito definendo un kernel CUDA per il
calcolo delle righe della matrice $V$. Poiché il kernel deve riempire
una riga per volta della matrice $V$, occorre usare _thread block_ in
una dimensione.

Compilare con:

        nvcc cuda-knapsack.cu -o cuda-knapsack -lm

Eseguire con:

        ./cuda-knapsack knap-100-100.in

## File

- [cuda-knapsack.cu](cuda-knapsack.cu)
- [hpc.h](hpc.h)
- [knapsack-gen.c](knapsack-gen.c)
- [knap-10-10.in](knap-10-10.in)
- [knap-100-100.in](knap-100-100.in)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* Problem instance */
typedef struct {
    int C;          /* capacity             */
    int n;          /* number of items      */
    int *w;         /* array of n weights   */
    float *v;       /* array of n values    */
} knapsack_t;

#ifndef SERIAL
__global__ void knapsack_first_row( float *P, int *w, float *v, int NCOLS )
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    if ( j < NCOLS )
	P[j] = (j < w[0] ? 0.0 : v[0]);
}

__device__ float cuda_fmaxf( float a, float b )
{
    return (a>b ? a : b);
}

__global__ void knapsack_step( float *Vcur, float *Vnext, int *w, float *v, int i, int NCOLS )
{
    const int j = threadIdx.x + blockIdx.x * blockDim.x;
    if ( j < NCOLS ) {
        Vnext[j] = Vcur[j];
        if ( j >= w[i] ) {
            Vnext[j] = cuda_fmaxf(Vcur[j], Vcur[j - w[i]] + v[i]);
        }
        /* Vnext[j] is the maximum profit that can be obtained by
           putting a subset of items {0, 1, ... i} into a container of
           capacity j */
    }
}

#define BLKSIZE 512

float knapsack(int C, int n, int* w, float *v)
{
    const int NCOLS = C+1;
    float *d_v;
    int *d_w;
    float *d_Vcur, *d_Vnext, *tmp;
    float result;
    int i;
    dim3 grid((NCOLS + BLKSIZE-1)/BLKSIZE);
    dim3 block(BLKSIZE);

    /* Allocate device copies of v and w */
    cudaSafeCall( cudaMalloc((void**)&d_w, n*sizeof(*d_w)) );
    cudaSafeCall( cudaMalloc((void**)&d_v, n*sizeof(*d_v)) );
    cudaSafeCall( cudaMalloc((void**)&d_Vcur, NCOLS*sizeof(*d_Vcur)) );
    cudaSafeCall( cudaMalloc((void**)&d_Vnext, NCOLS*sizeof(*d_Vnext)) );

    /* Copy input data */
    cudaSafeCall( cudaMemcpy(d_w, w, n*sizeof(*w), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_v, v, n*sizeof(*v), cudaMemcpyHostToDevice) );

    knapsack_first_row<<<grid, block>>>(d_Vcur, d_w, d_v, NCOLS);
    cudaCheckError();

    /* Compute the DP matrix row-wise */
    for (i=1; i<n; i++) {
        knapsack_step<<<grid, block>>>(d_Vcur, d_Vnext, d_w, d_v, i, NCOLS);
        cudaCheckError();

        tmp = d_Vcur;
        d_Vcur = d_Vnext;
        d_Vnext = tmp;
    }

    cudaSafeCall( cudaMemcpy(&result, &d_Vcur[NCOLS-1], sizeof(result), cudaMemcpyDeviceToHost) );

    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_Vcur);
    cudaFree(d_Vnext);
    return result;
}

#else

/**
 * Given a set of n objects of weights w[0], ... w[n-1] and values
 * v[0], ... v[n-1], compute the maximum profit that can be obtained
 * by putting a subset of objects into a container of total capacity
 * C. Formally, the goal is to find a binary vector x[0], ... x[n-1]
 * such that:
 *
 * sum_{i=0}^{n-1} x[i] * v[i] is maximized
 *
 * subject to: sum_{i=0}^{n-1} x[i] * w[i] <= C
 *
 * This function uses the standard approach based on dynamic
 * programming; therefore, it requires space proportional to n*C
 */
float knapsack(int C, int n, int* w, float *v)
{
    const int NROWS = n;
    const int NCOLS = C+1;
    float *Vcur, *Vnext, *tmp;
    float result;
    int i, j;

    /* [TODO] questi array andranno allocati nella memoria del device */
    Vcur = (float*)malloc(NCOLS*sizeof(*Vcur));
    assert(Vcur != NULL);

    Vnext = (float*)malloc(NCOLS*sizeof(*Vnext));
    assert(Vnext != NULL);

    /* Inizializzazione: [TODO] volendo si puo' trasformare questo
       ciclo in un kernel CUDA */
    for (j=0; j<NCOLS; j++) {
	Vcur[j] = (j < w[0] ? 0.0 : v[0]);
    }
    /* Compute the DP matrix row-wise */
    for (i=1; i<NROWS; i++) {
        /* [TODO] Scrivere un kernel che esegua il ciclo seguente
           eseguendo NCOLS thread in parallelo */
        for (j=0; j<NCOLS; j++) {
	    Vnext[j] = Vcur[j];
            if ( j>=w[i] ) {
                Vnext[j] = fmaxf(Vcur[j], Vcur[j - w[i]] + v[i]);
            }
        }
        /* Vnext[j] is the maximum profit that can be obtained by
           putting a subset of items {0, 1, ... i} into a container of
           capacity j */
        tmp = Vcur;
        Vcur = Vnext;
        Vnext = tmp;
    }

    result = Vcur[NCOLS-1];
    free(Vcur);
    free(Vnext);
    return result;
}
#endif

/* Read and allocate a problem instance from file |fin|; the file must
   conta, in order, C n w v. The problem instance can be deallocated
   with knapsack_free() */
void knapsack_load(FILE *fin, knapsack_t* k)
{
    int i;
    assert(k != NULL);
    fscanf(fin, "%d", &(k->C)); assert( k->C > 0 );
    fscanf(fin, "%d", &(k->n)); assert( k->n > 0 );
    k->w = (int*)malloc((k->n)*sizeof(int)); assert(k->w != NULL);
    k->v = (float*)malloc((k->n)*sizeof(float)); assert(k->v != NULL);
    for (i=0; i<(k->n); i++) {
        const int nread = fscanf(fin, "%d %f", k->w + i, k->v + i);
        assert(2 == nread);
	assert( k->w[i] >= 0 );
	assert( k->v[i] >= 0 );
        /* printf("%d %f\n", *(k->w + i), *(k->v + i)); */
    }
    printf("Loaded knapsack instance with %d items, capacity %d\n", k->n, k->C);
}

/* Deallocate all memory used by a problem instance */
void knapsack_free(knapsack_t* k)
{
    assert(k != NULL);
    k->n = k->C = 0;
    free(k->w);
    free(k->v);
}

void knapsack_solve(const knapsack_t* k)
{
    const float result = knapsack(k->C, k->n, k->w, k->v);
    printf("Optimal profit: %f\n", result);
}

int main(int argc, char* argv[])
{
    knapsack_t k;
    FILE *fin;
    if ( 2 != argc ) {
        fprintf(stderr, "Usage: %s inputfile\n", argv[0]);
        return EXIT_FAILURE;
    }
    fin = fopen(argv[1], "r");
    if (NULL == fin) {
        fprintf(stderr, "Can not open \"%s\" for reading\n", argv[1]);
        return EXIT_FAILURE;
    }
    knapsack_load(fin, &k);
    fclose(fin);
    knapsack_solve(&k);
    knapsack_free(&k);
    return EXIT_SUCCESS;
}
