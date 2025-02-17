/****************************************************************************
 *
 * opencl-knapsack.c - 0-1 Knapsack problem
 *
 * Copyright (C) 2017--2023 Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla/>
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
% HPC - 0-1 Knapsack problem
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla/)
% Last updated: 2023-10-24

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

1. If $w_i > j$, the weight of item $i$ exceeds by itself the capacity
   of the knapsack, so that item $i$ can definitely not be used.
   Therefore, the optimal solution $V(i,j)$ of problem $P(i,j)$ will
   not contain item $i$, and will then be the same as the optimal
   solution $V(i-1,j)$ of problem $P(i-1,j)$.

2. If $w_i \leq j$, then we may or may not use item $i$. The choice
   depends on which alternative provides the better outcome:

    a. If we choose to use item $i$, then the solution $V(i,j)$ of
       problem $P(i,j)$ is $V(i-1,j-w_i)+v_i$: indeed, we use item $i$
       of value $v_i$, and fill the residual capacity $j - w_i$ with
       the items chosen among $\{0, 1, \ldots, i-1\}$ that provide the
       maximum value. Such maximum value is the solution $V(i-1,
       j-w_i)$ of $P(i-1, j-w_i)$.

    b. If we choose not to use item $i$, the maximum value that we can
       put into the knapsack is $V(i-1, j)$ as in case 1 above.

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
keep track of which items belong to the optimal solution.  For the
sake of simplicity, in this exercise we only compute the value of the
optimal solution.

The file [opencl-knapsack.c](opencl-knapsack.c) contains a serial
solution to the 0-1 Knapsack problem using dynamic programming. Input
instances are read from a text file whose name is provided as the only
command-line parameter. The input file has a simple structure:

- The first value is the maximum capacity $C$ (integer);

- The second value is the number of items $n$ (integer);

- After that, $n$ pairs $(w_i, v_i)$ are listed, separated by spaces or tabs.

The program [knapsack-gen.c](knapsack-gen.c) generates random
instances for additional experiments.

The goal is to modify the program to make use of OpenCL
parallelism. You are suggested to write a suitable kernel that fills a
row of matrix $V$ concurrently; the CPU will then run the kernel for
each row, from top to bottom, until the result is computed. Since the
domain is 1D (essentially, an array), you should organize work-items
in 1D workgroups as well.

Compile with:

        cc opencl-knapsack.c simpleCL.c -o opencl-knapsack -lm -lOpenCL

Run with:

        ./opencl-knapsack knap-100-100.in

## Files

- [opencl-knapsack.c](opencl-knapsack.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpelCL.h)
- [knapsack-gen.c](knapsack-gen.c) (to generate random input instances)
- [knap-10-10.in](knap-10-10.in)
- [knap-100-100.in](knap-100-100.in)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "simpleCL.h"

/* Problem instance */
typedef struct {
    int C;          /* capacity             */
    int n;          /* number of items      */
    int *w;         /* array of n weights   */
    float *v;       /* array of n values    */
} knapsack_t;

#ifndef SERIAL
sclKernel knapsack_first_row;
sclKernel knapsack_step;

float knapsack(knapsack_t *k)
{
    /* Let us give shorter names to frequently used attributes of
       structure `k` */
    const int C = k->C;
    const int n = k->n;
    int *w = k->w;
    float *v = k->v;
    const int NCOLS = C+1;
    cl_mem d_v, d_w, d_Vcur, d_Vnext, tmp;
    const sclDim grid = DIM1(sclRoundUp(NCOLS, SCL_DEFAULT_WG_SIZE));
    const sclDim block = DIM1(SCL_DEFAULT_WG_SIZE);

    /* Allocate device copies of v and w */
    d_w = sclMallocCopy(n*sizeof(*w), w, CL_MEM_READ_ONLY);
    d_v = sclMallocCopy(n*sizeof(*v), v, CL_MEM_READ_ONLY);
    d_Vcur = sclMalloc(NCOLS*sizeof(*v), CL_MEM_READ_WRITE);
    d_Vnext = sclMalloc(NCOLS*sizeof(*v), CL_MEM_READ_WRITE);

    sclSetArgsEnqueueKernel(knapsack_first_row,
                            grid, block,
                            ":b :b :b :d",
                            d_Vcur, d_w, d_v, NCOLS);

    /* Compute the DP matrix row-wise */
    for (int i=1; i<n; i++) {
        sclSetArgsLaunchKernel(knapsack_step,
                               grid, block,
                               ":b :b :b :b :d :d",
                               d_Vcur, d_Vnext, d_w, d_v, i, NCOLS);
        tmp = d_Vcur;
        d_Vcur = d_Vnext;
        d_Vnext = tmp;
    }

    /* To fetch the last element of d_Vcur, we need to use
       sclMemcpyDeviceToHostOFfset() */
    float result;
    sclMemcpyDeviceToHostOffset(&result, d_Vcur, (NCOLS-1)*sizeof(*v), sizeof(result));

    sclFree(d_v);
    sclFree(d_w);
    sclFree(d_Vcur);
    sclFree(d_Vnext);
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
float knapsack(knapsack_t *k)
{
    /* Let us give shorter names to frequently used attributes of
       structure `k` */
    const int C = k->C;
    const int n = k->n;
    int *w = k->w;
    float *v = k->v;
    const int NROWS = n;
    const int NCOLS = C+1;
    float *Vcur, *Vnext, *tmp;
    float result;

    /* [TODO] allocate `Vcur` and `Vnext` on device memory */
    Vcur = (float*)malloc(NCOLS*sizeof(*Vcur));
    assert(Vcur != NULL);

    Vnext = (float*)malloc(NCOLS*sizeof(*Vnext));
    assert(Vnext != NULL);

    /* Initialization (this could be transformed into an OpenCL
       kernel, but is probably not worth the effort) */
    for (int j=0; j<NCOLS; j++) {
	Vcur[j] = (j < w[0] ? 0.0 : v[0]);
    }
    /* Compute the DP matrix row-wise */
    for (int i=1; i<NROWS; i++) {
        /* [TODO] Rewrite the loop below as a kernel launch using
           NCOLS work-items */
        for (int j=0; j<NCOLS; j++) {
            if ( j >= w[i] ) {
                Vnext[j] = fmaxf(Vcur[j], Vcur[j - w[i]] + v[i]);
            } else {
                Vnext[j] = Vcur[j];
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

/* Read and allocate a problem instance from file `fin`; the file must
   contain `C` `n` `w` `v`, in the given order. The problem instance
   can be deallocated with `knapsack_free()` */
void knapsack_load(FILE *fin, knapsack_t* k)
{
    assert(fin != NULL);
    assert(k != NULL);
    fscanf(fin, "%d", &(k->C)); assert( k->C > 0 );
    fscanf(fin, "%d", &(k->n)); assert( k->n > 0 );
    k->w = (int*)malloc((k->n)*sizeof(int)); assert(k->w != NULL);
    k->v = (float*)malloc((k->n)*sizeof(float)); assert(k->v != NULL);
    for (int i=0; i<(k->n); i++) {
        const int nread = fscanf(fin, "%d %f", k->w + i, k->v + i);
        assert(2 == nread);
	assert( k->w[i] >= 0 );
	assert( k->v[i] >= 0 );
        /* printf("%d %f\n", *(k->w + i), *(k->v + i)); */
    }
    fprintf(stderr, "Loaded knapsack instance with %d items, capacity %d\n", k->n, k->C);
}

/* Deallocate all memory used by a problem instance */
void knapsack_free(knapsack_t* k)
{
    assert(k != NULL);
    k->n = k->C = 0;
    free(k->w);
    free(k->v);
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
#ifndef SERIAL
    sclInitFromFile("opencl-knapsack.cl");
    knapsack_first_row = sclCreateKernel("knapsack_first_row");
    knapsack_step = sclCreateKernel("knapsack_step");
#endif
    knapsack_load(fin, &k);
    fclose(fin);
    const float result = knapsack(&k);
    printf("Optimal profit: %f\n", result);
    knapsack_free(&k);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
