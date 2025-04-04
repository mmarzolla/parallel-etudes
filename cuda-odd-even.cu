/****************************************************************************
 *
 * cuda-odd-even.cu - Odd-even sort
 *
 * Copyright (C) 2017--2024 Moreno Marzolla
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
% HPC - Odd-even sort
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last upated: 2025-03-22

The _Odd-Even sort_ algorithm is a variant of BubbleSort, and sorts an
array of $n$ elements in $O(n^2)$ sequential time. Although not
efficient, odd-even sort is easily parallelizable. The goal of this
exercise is to write a CUDA version of Odd-Even sort.

Given an array `v[]` of length $n$, the algorithm performs $n$ steps
numbered $0, \ldots, n-1$. During even steps, array elements in even
positions are compared with their successors and swapped if not in the
correct order. During odd steps, elements in odd position are compared
(and possibly swapped) with their successors. See Figure 1.

![Figure 1: Odd-Even Sort](cuda-odd-even.svg)

The file [cuda-odd-even.cu](cuda-odd-even.cu) contains a serial
implementation of Odd-Even transposition sort. The purpose of this
exercise is to modify the program to use the GPU.

The CUDA paradigm suggests a fine-grained parallelism where a CUDA
thread is responsible for a single compare-and-swap operation. The
simplest solution is to launch $n$ CUDA threads during each phase;
only even (resp. odd) threads will be active during even (resp. odd)
phases. The kernel looks like this:

```C
__global__ void odd_even_step_bad( int *x, int n, int phase )
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( (idx < n-1) && ((idx % 2) == (phase % 2)) ) {
		cmp_and_swap(&x[idx], &x[idx+1]);
	}
}
```

This solution is simple but not efficient because only half the
threads are active during each phase, so a lot of computational
resources are wasted. To address this issue, write a second version
where $\lceil n/2 \rceil$ CUDA threads are launched during each
phase. Indexing becomes more problematic, since each thread should be
uniquely assigned to an even (resp. odd) position depending on the
phase. Specifically, during even phases, threads $0, 1, 2, 3, \ldots$
must handle the pairs $(0, 1)$, $(2, 3)$, $(4, 5)$, $(6, 7)$,
$\ldots$. During odd phases, the threads must handle the pairs $(1,
2)$, $(3, 4)$, $(5, 6)$, $(7, 8)$, $\ldots$.

Table 1 illustrates the mapping between the linear thread index `idx`
and the pair of elements it needs to manage.

:Table 1: Mapping thread index to array index pairs.

Thread index       Even phases   Odd phases
-----------------  ------------  --------------
0                  $(0,1)$       $(1,2)$
1                  $(2,3)$       $(3,4)$
2                  $(4,5)$       $(5,6)$
3                  $(6,7)$       $(7,8)$
4                  $(8,9)$       $(9,10)$
...                ...           ...
-----------------  ------------  --------------

To compile:

        nvcc cuda-odd-even.cu -o cuda-odd-even

To execute:

        ./cuda-odd-even [len]

Example:

        ./cuda-odd-even 1024

## Files

- [cuda-odd-even.cu](cuda-odd-even.cu)
- [hpc.h](hpc.h)

 ***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

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
    for (int i=0; i<n; i++) {
        x[i] = i;
    }
    for(int i=0; i<n-1; i++) {
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
    for (int i=0; i<n; i++) {
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
