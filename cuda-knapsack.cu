/****************************************************************************
 *
 * cuda-knapsack.c - 0-1 knapsack problem
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - 0-1 knapsack problem
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-10-01

![](knapsack.png)

The 0/1 Knapsack problem is a well-known optimization problem that is
part of the general family of _Knapsack problems_. The input consists
of $n$ items of positive integer weights $w_0, \ldots, w_{n-1}$ and
real, nonnegative values $v_0, \ldots, v_{n-1}$. We are also given a
knapsack that can hold any number of items, provided that the total
weight does not exceed a given capacity $C$.  The goal is to identify
the subset of items of maximum total values whose total weight is less
than or equal to $C$. The following explanation of the algorithm is
not required to solve this exercise; however, I suggest you to read it
to grasp at least the main idea behind the code.

This problem can be solved using dynamic programming. Let $P(i,j)$,
$i=0, \ldots, (n-1)$, $j=0, \ldots, C$ be a family of problems defined
as follows:

> $P(i,j)$ consists of finding the subset of items chosen among $\{0,
> \ldots, i\}$ whose total weight is less than or equal to $j$ and
> whose total value is maximum possible.

The solution of the 0-1 Knapsack problem is then the solution of
$P(n-1, C)$.

Let $V(i,j)$ the maximum value of a subset of items $\{0, \ldots, i\}$
whose total weight is less than or equal to $j$ (in other word,
let $V(i,j)$ be the solution of problem $P(i,j)$). 

We know that $V(i,0) = 0$; indeed, in a container of zero capacity no
item can be inserted.

Regarding $V(0,j)$, only the first item with weight $w_0$ and value
$v_0$ is available. Therefore, the only possibilities is to insert the
item into the knapsack or not. If the capacity of the knapsack is at
least $j$, the maximum value can be obtained by inserting the
item. Otherwise, the maximum value is zero. Therefore, we can write:

$$
V(0,j) = \begin{cases}
0 & \mbox{if}\ j < w_0\\
v_0 & \mbox{otherwise}
\end{cases}
$$

The general case is a bit tricky. The solution of $P(i,j)$ may or may
not use item $i$. We have the following cases:

1. If $w_i >j$, the weight of item $i$ exceeds by itself the capacity
   of the knapsack, so that item $i$ can definitely not be used.
   Therefore, the optimal solution $V(i,j)$ of problem $P(i,j)$ will
   not contain item $i$, and will then be the same as the optimal
   solution $V(i-1,j)$ of problem $P(i-1,j)$.

2. If $w_i \leq j$, then we may or may not use item $i$. The choice
   depends on which alternative provides the better value.

    a. If we choose to use item $i$, then the optimal solution
       $V(i,j)$ of problem $P(i,j)$ is $V(i-1,j-w_i)+v_i$: in fact, we
       use item $i$ of value $v_i$, and we fill the residual capacity
       $j - w_i$ of the knapsack with the items chosen among the
       remaining $\{9, 1, \ldost,k i-1\]$ that provide the maximum
       value. Such maximum value is precisely the solution $V(i-1,
       j-w_i)$ of problem $P(i-1, j-w_i)$.

    b. If we choose not to use item $i$, the maximum value that we can
       insert into the knapsack is $V(i-1, j)$ as in case 1 above.

So, should be use item $i$ or not? We choose the alternative among 2.a
and 2.b that maximizes the total value.  Therefore, for any $i=1,
\ldots, (n-1)$, $j=0, \ldots, C$ we write:

$$
V(i,j) = \begin{cases}
V(i-1, j) & \mbox{if}\ j < w_i\\
\max\{V(i-1, j), V(i-1, j-w_i) + v_i\} & \mbox{otherwise}
\end{cases}
$$

With a slight modification of the algorithm above it is possible to
keep track of _which_ items belong to the optimal solution.  For the
sake of simplicity, in this exercise we want to compute only the value
of the optimal solution.

The file [cuda-knapsack.cu](cuda-knapsack.cu) solves the problem using
the CPU only. The program reads a problem instance from an input file
whose name is passed on the command line; at the end, the maximum
total value of the objects that can be inserted into the knapsack is
printed to standard output.  The input file has a simple structure:
the first two lines contain the values $C$ and $n$; $n$ rows follow,
each containing the integer weight $w_i$ and real value $v_i$ of
object $i$. The program [knapsack-gen.c](knapsack-gen.c) can be used
to generate a random input file.

You are required to modify the program and define suitable CUDA
kernels to compute the rows of matrix $V$. Due to data dependences,
only the values on the same row of $V$ can be computed concurrently.
Therefore, you should define a 1D thread block and map threads to a
row of $V$. The kernel should be invoked for each row.

To compile:

        nvcc cuda-knapsack.cu -o cuda-knapsack -lm

To execute:

        ./cuda-knapsack knap-100-100.in

## Files

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

#define BLKDIM 512

float knapsack(int C, int n, int* w, float *v)
{
    const int NCOLS = C+1;
    float *d_v;
    int *d_w;
    float *d_Vcur, *d_Vnext, *tmp;
    float result;
    int i;
    dim3 grid((NCOLS + BLKDIM-1)/BLKDIM);
    dim3 block(BLKDIM);

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
