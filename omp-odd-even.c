/****************************************************************************
 *
 * omp-odd-even.c - Odd-even transposition sort using OpenMP
 *
 * Last modified in 2025 by Moreno Marzolla <https://www.unibo.it/sitoweb/moreno.marzolla>
 *
 * The original copyright notice follows.
 *
 * --------------------------------------------------------------------------
 *
 * Copyright (c) 2000, 2013, Peter Pacheco and the University of San
 * Francisco. All rights reserved. Redistribution and use in source
 * and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***************************************************************************/

/***
% Odd-even sort
% [Peter Pacheco](https://www.cs.usfca.edu/~peter/), [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last modified: 2026-06-09

The _Odd-Even sort_ algorithm is a variant of BubbleSort, and sorts an
array of $n$ elements in $O(n^2)$ sequential time. Although not
efficient, odd-even sort is easily parallelizable. The goal of this
exercise is to write a parallel version of Odd-Even sort using OpenMP.

Given an array `v[]` of length $n$, the algorithm performs $n$ steps
numbered $0, \ldots, n-1$. During even steps, array elements in even
positions are compared with their successors and swapped if not in the
correct order. During odd steps, elements in odd position are compared
(and possibly swapped) with their successors. See Figure 1.

![Figure 1: Odd-Even Sort](cuda-odd-even.svg)

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-odd-even.c -o cuda-odd-even

To execute:

        ./omp-odd-even [len]

Example:

        ./omp-odd-even 1024

## Files

- [omp-odd-even.c](omp-odd-even.c)

***/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
    }
}

/* Fills vector v with a permutation of the integer values 0, .. n-1 */
void fill( int* v, int n )
{
    int up = n-1, down = 0;
    for ( int i=0; i<n; i++ ) {
	v[i] = ( i % 2 == 0 ? up-- : down++ );
    }
}

void odd_even_sort( int* v, int n )
{
#ifndef SERIAL
#pragma omp parallel default(none) shared(n,v)
#endif
    for (int phase = 0; phase < n; phase++) { /* note: `phase` is private to each thread */
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
#ifndef SERIAL
#pragma omp for
#endif
	    for (int i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
#ifndef SERIAL
#pragma omp for
#endif
	    for (int i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

void check( const int* v, int n )
{
    for (int i=0; i<n-1; i++) {
	if ( v[i] != i ) {
	    printf("Check FAILED: v[%d]=%d, expected %d\n",
		   i, v[i], i );
	    abort();
	}
    }
    printf("Check OK\n");
}

typedef void (* odd_even_sort_t)(int *, int);

void test(const char* desc, odd_even_sort_t f, int *v, int n)
{
    const int NREPS = 5;

    printf("%s (array len=%d, replications=%d)\n", desc, n, NREPS);
    const double tstart = omp_get_wtime();
    for (int r=0; r<NREPS; r++) {
        printf("Run %d of %d\n", r+1, NREPS);
        f(v,n);
    }
    const double elapsed = omp_get_wtime() - tstart;
    printf("Average elapsed time %.3f\n", elapsed/NREPS);
}

int main( int argc, char* argv[] )
{
    int n = 100000;
    int *v;

    if ( argc > 1 ) {
	n = atoi(argv[1]);
    }
    v = (int*)malloc(n*sizeof(v[0]));
    fill(v,n);
    printf("Sorting %d elements\n", n);
    const double tstart = omp_get_wtime();
    odd_even_sort(v, n);
    const double elapsed = omp_get_wtime() - tstart;
    printf("Execution time: %.3f\n", elapsed);
    check(v,n);
    free(v);
    return EXIT_SUCCESS;
}
