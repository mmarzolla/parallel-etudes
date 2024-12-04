/****************************************************************************
 *
 * omp-loop.c - Loop-carried dependences
 *
 * Copyright (C) 2018--2022, 2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Loop-carried dependences
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-10-11

The file [omp-loop.c](omp-loop.c) contains a set of serial functions
with loops that iterate over arrays or matrices. The goal of this
exercise is to apply the loop parallelization techniques seen during
the class (or according to the hint provided below) to create a
parallel version.

The `main()` function checks for correctness of the results comparing
the output of the serial and parallel versions. Note that such fact
check is not (and can not be) exhaustive.

To compile:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-loop.c -o omp-loop

To execute:

        ./omp-loop

## Files

- [omp-loop.c](omp-loop.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

/* Three small functions used below; you should not need to know what
   these functions do */
int f(int a, int b, int c) { return (a+b+c)/3; }
int g(int a, int b) { return (a+b)/2; }
int h(int a) { return (a > 10 ? 2*a : a-1); }

/****************************************************************************/

/**
 * Shift the elements of array |a| of length |n| one position to the
 * right; the rightmost element of |a| becomes the leftmost element of
 * the shifted array.
 */
void vec_shift_right_seq(int *a, int n)
{
    const int tmp = a[n-1];
    for (int i=n-1; i>0; i--) {
        a[i] = a[i-1];
    }
    a[0] = tmp;
}

void vec_shift_right_par1(int *a, int n)
{
    /* Parallel version of `vec_shift_right_seq()`. It is not possible
       to remove the loop-carried dependence by aligning loop
       iterations. I suggest to use a temporary array `b[]`, and split
       the loop into two loops: the first copies all elements of `a[]`
       in the shifted position of `b[]` (i.e., `a[i]` goes to
       `b[i+1]`; the rightmost element of `a[]` goes into `b[0]`). The
       second loop copies `b[]` into `a[]`. Both loops can be
       trivially parallelized. */
#ifdef SERIAL
    /* TODO */
#else
    int *b = (int*)malloc(n*sizeof(b[0]));
    assert(b != NULL);
#pragma omp parallel default(none) shared(a,b,n)
    {
#pragma omp for
        for (int i=1; i<n; i++) {
            b[i] = a[i-1];
        } /* Implicit synchronization here */
#pragma omp single
        b[0] = a[n-1];
        /* Implicit synchronization here, after "omp single" */
#pragma omp for
        for (int i=0; i<n; i++) {
            a[i] = b[i];
        } /* Implicit synchronization here */
    }
    free(b);
#endif
}

void vec_shift_right_par2(int *a, int n)
{
    /* A different solution to shift a vector without using a
       temporary array: partition `a[]` into P blocks (P=size of the
       team). Each process saves the rightmost element of its block
       into a shared array of length P, and then shifts the block on
       position right. When all threads are done (barrier
       synchronization), each thread fills the _leftmost_ element of
       its block with the _rightmost_ element saved by its left
       neighbor.

       Example, with P=4 threads:

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

       Each thread stores the rightmost element into a shared array
       rightmost[]:

       +-+-+-+-+
       |f|l|r|x|   rightmost[]
       +-+-+-+-+

       Each thread shifts right its portion; the leftmost element of
       each partition may have any value (?) and will be overwritten
       in the next step

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |?|a|c|b|d|e|?|g|h|i|j|k|?|m|n|o|p|q|?|s|t|u|v|w|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

       Each thread fills the leftmost element with the correct value
       from `rightmost[]`

       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       |x|a|c|b|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|
       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
       \----------/\----------/\----------/\----------/
            P0          P1          P2          P3

    */
#ifndef SERIAL
    const int num_threads = omp_get_max_threads();
    int rightmost[num_threads];
#pragma omp parallel default(none) shared(num_threads,rightmost,a,n)
    {
        const int my_id = omp_get_thread_num();
        const int my_start = n * my_id / num_threads;
        const int my_end = n * (my_id + 1) / num_threads;
        const int left_neighbor = (my_id > 0 ? my_id - 1 : num_threads - 1);

        rightmost[my_id] = a[my_end - 1];
        for (int i = my_end - 1; i > my_start; i--) {
            a[i] = a[i-1];
        }
#pragma omp barrier
        a[my_start] = rightmost[left_neighbor];
    }
#endif
}

/****************************************************************************/

/* This function converts 2D indexes into a linear index; n is the
   number of columns of the matrix being indexed. A better solution
   would be to use C99-style casts:

        int (*AA)[n] = (int (*)[n])A;

   and then write `AA[i][j]`. Unfortunately, this triggers a bug in
   gcc 5.4.0+OpenMP (works with gcc 8.2.0+OpenMP)
*/
int IDX(int i, int j, int n)
{
    return i*n + j;
}

/* A is a nxn matrix */
void test1_seq(int *A, int n)
{
    for (int i=1; i<n; i++) {
        for (int j=1; j<n-1; j++) {
            /*
              A[i][j] = f(A[i-1][j-1], A[i-1][j], A[i-1][j+1])
            */
            A[IDX(i,j,n)] = f(A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)], A[IDX(i-1,j+1,n)]);
        }
    }
}

void test1_par(int *A, int n)
{
#ifdef SERIAL
    /* [TODO] This function should be a parallel vesion of
       test1_seq(). Suggestion: start by drawing the dependences among
       the elements of matrix A[][] as they are computed.  Then,
       observe that one of the loops (which one?) can be parallelized
       with a "#pragma omp parallel for" directive. There is no need
       to modify the code, nor to exchange loops. */
#else
    for (int i=1; i<n; i++) {
#pragma omp parallel for default(none) shared(A,i,n)
        for (int j=1; j<n-1; j++) {
            A[IDX(i,j,n)] = f(A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)], A[IDX(i-1,j+1,n)]);
        }
    }
#endif
}

/****************************************************************************/

void test2_seq(int *A, int n)
{
    for (int i=1; i<n; i++) {
        for (int j=1; j<n; j++) {
            /*
              A[i][j] = g(A[i,j-1], A[i-1,j-1])
             */
            A[IDX(i,j,n)] = g(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)]);
        }
    }
}

void test2_par(int *A, int n)
{
#ifdef SERIAL
    /* [TODO] This function should be a parallel version of
       `test2_seq()`. Suggestion: start by drawing the dependences
       among the elements of matrix `A[][]` as they are
       computed. Observe that it is not possible to put a "parallel
       for" directive on either loop.

       However, you can exchange the loops (why?), i.e., the loops
       can be rewritten as

       for (int j=1; j<n; j++) {
         for (int i=1; i<n; i+) {
           ....
         }
       }

       preserving the correctness of the computation. Now, one of the
       loops can be parallelized (which one?) */
#else
    /* Loop interchange */
    for (int j=1; j<n; j++) {
#pragma omp parallel for default(none) shared(A,j,n)
        for (int i=1; i<n; i++) {
            A[IDX(i,j,n)] = g(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)]);
        }
    }
#endif
}

/****************************************************************************/

void test3_seq(int *A, int n)
{
    for (int i=1; i<n; i++) {
        for (int j=1; j<n; j++) {
            /*
              A[i][j] = f(A[i][j-1], A[i-1][j-1], A[i-1][j])
             */
            A[IDX(i,j,n)] = f(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)]);
        }
    }
}

void test3_par(int *A, int n)
{
#ifdef SERIAL
    /* [TODO] This function should be a parallel version of
       `test3_seq()`. Neither loops can be trivially parallelized;
       exchanging loops (moving the inner loop outside) does not work
       either.

       This is the same example shown on the slides, and can be solved
       by sweeping the matrix "diagonally".

       There is a caveat: the code on the slides sweeps the _whole_
       matrix; in other words, variables i and j will assume all
       values starting from 0. The code of `test3_seq()` only process
       indexes where i>0 and j>0, so you need to add an "if" statement
       to skip the case where i==0 or j==0. */
#else
    for (int slice=0; slice < 2*n - 1; slice++) {
	const int z = slice < n ? 0 : slice - n + 1;
#pragma omp parallel for default(none) shared(A, n, slice, z)
	for (int i = slice - z; i >= z; i--) {
            const int j = slice - i;
            if (i>0 && j>0) {
                A[IDX(i,j,n)] = f(A[IDX(i,j-1,n)], A[IDX(i-1,j-1,n)], A[IDX(i-1,j,n)]);
            }
	}
    }
#endif
}

/**
 ** The code below does not need to be modified
 **/

void fill(int *a, int n)
{
    a[0] = 31;
    for (int i=1; i<n; i++) {
        a[i] = (a[i-1] * 33 + 1) % 65535;
    }
}

int array_equal(int *a, int *b, int n)
{
    for (int i=0; i<n; i++) {
        if (a[i] != b[i]) { return 0; }
    }
    return 1;
}

int main( void )
{
    const int N = 1024;
    int *a1, *b1, *c1, *a2, *b2, *c2;

    /* Allocate enough space for all tests */
    a1 = (int*)malloc(N*N*sizeof(int)); assert(a1 != NULL);
    b1 = (int*)malloc(N*sizeof(int)); assert(b1 != NULL);
    c1 = (int*)malloc(N*sizeof(int)); assert(c1 != NULL);

    a2 = (int*)malloc(N*N*sizeof(int)); assert(a2 != NULL);
    b2 = (int*)malloc(N*sizeof(int)); assert(b2 != NULL);
    c2 = (int*)malloc(N*sizeof(int)); assert(c2 != NULL);

    printf("vec_shift_right_par1()\t"); fflush(stdout);
    fill(a1, N);
    vec_shift_right_seq(a1, N);
    fill(a2, N);
    vec_shift_right_par1(a2, N);
    if ( array_equal(a1, a2, N) ) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    printf("vec_shift_right_par2()\t"); fflush(stdout);
    fill(a2, N);
    vec_shift_right_par2(a2, N);
    if ( array_equal(a1, a2, N) ) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test1 */
    printf("test1_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test1_seq(a1, N);
    fill(a2, N*N);
    test1_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test2 */
    printf("test2_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test2_seq(a1, N);
    fill(a2, N*N);
    test2_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    /* test3 */
    printf("test3_par()\t\t"); fflush(stdout);
    fill(a1, N*N);
    test3_seq(a1, N);
    fill(a2, N*N);
    test3_par(a2, N);
    if (array_equal(a1, a2, N*N)) {
        printf("OK\n");
    } else {
        printf("FAILED\n");
    }

    free(a1);
    free(b1);
    free(c1);
    free(a2);
    free(b2);
    free(c2);

    return EXIT_SUCCESS;
}
