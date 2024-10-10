/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Array reversal with CUDA
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-11-29

Write a program that reverses an array `v[]` of length $n$, i.e.,
exchanges `v[0]` and `v[n-1]`, `v[1]` and `v[n-2]` and so on. You
should write the following functions:

1. `reverse()` reverses an array `in[]` into a different array `out[]`
   (the input is not modified). Assume that `in[]` and `out[]` reside
   on non-overlapping memory blocks.

2. `inplace_reverse()` reverses the array `in[]` "in place", i.e.,
   exchanging elements using $O(1)$ additional storage; therefore, you
   are not allowed to allocate a temporary output vector.

The file [cuda-reverse.cu](cuda-reverse.cu) provides a serial
implementation of `reverse()` and `inplace_reverse()`. Your goal is to
odify the functions to use of the GPU, defining any additional kernel
that is required.

## Hints

`reverse()` can be parallelized by launching $n$ CUDA threads; each
thread copies a single element from the input to the output
array. Since the array size $n$ can be large, you should create as
many one-dimensional _thread blocks_ as needed to have at least $n$
threads. Have a look at the lecture notes on how to do this.

`inplace_reverse()` can be parallelized by launching $\lfloor n/2
\rfloor$ CUDA threads (note the rounding): each thread swaps an
element on the first half of `in[]` with the corresponding element on
the second half.

To map threads to array elements it is possible to use the expression:

```C
const int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

In both cases the program might create more threads than actually
needed; special care should be made to ensure that the extra threads
do nothing, e.g., using

```C
if (idx < n) {
  \/\* body \*\/
}
\/\* else do nothing \*\/
```

for `reverse()`, and

```C
if (idx < n/2) {
  \/\* body \*\/
}
\/\* else do nothing \*\/
```

for `inplace_reverse()`.

To compile:

        nvcc cuda-reverse.cu -o cuda-reverse

To execute:

        ./cuda-reverse [n]

Example:

        ./cuda-reverse

## Files

- [cuda-reverse.cu](cuda-reverse.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef SERIAL
#define BLKDIM 1024

/* Reverses `in[]` into `out[]`. Assume that `in[]` and `out[]` always
   point to non-overlapping memory blocks. Uses `n` CUDA threads to
   reverse `n` elements */
__global__ void reverse_kernel( int *in, int *out, int n )
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n ) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
}
#endif

/* Reverses `in[]` into `out[]`; assume that `in[]` and `out[]` do not
   overlap.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in` and `out`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
 */
void reverse( int *in, int *out, int n )
{
#ifdef SERIAL
    for (int i=0; i<n; i++) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
#else
    int *d_in, *d_out; /* device copy of `in` and `out` */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] and out[] */
    cudaSafeCall( cudaMalloc((void **)&d_in, SIZE) );
    cudaSafeCall( cudaMalloc((void **)&d_out, SIZE) );

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );

    /* Launch `reverse_kernel()` on the GPU */
    reverse_kernel<<<(n + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_in, d_out, n);
    cudaCheckError();

    /* Copy the result back to host memory */
    cudaSafeCall( cudaMemcpy(out, d_out, SIZE, cudaMemcpyDeviceToHost) );

    /* Free memory on the device */
    cudaFree(d_in);
    cudaFree(d_out);
#endif
}

#ifndef SERIAL
/* In-place reversal of `in[]`; n/2 CUDA threads are required to
   reverse n elements */
__global__ void inplace_reverse_kernel( int *in, int n )
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n/2 ) {
        const int opp = n - 1 - i;
        const int tmp = in[opp];
        in[opp] = in[i];
        in[i] = tmp;
    }
}
#endif

/* In-place reversal of in[] into itself.

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in`;
   - copies `in` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
*/
void inplace_reverse( int *in, int n )
{
#ifdef SERIAL
    int i = 0, j = n-1;
    while (i < j) {
        const int tmp = in[j];
        in[j] = in[i];
        in[i] = tmp;
        j--;
        i++;
    }
#else
    int *d_in; /* device copy of in */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of `in[]` */
    cudaSafeCall( cudaMalloc((void **)&d_in, SIZE) );

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );

    /* Launch `reverse_kernel()` on the GPU */
    inplace_reverse_kernel<<<(n/2 + BLKDIM - 1)/BLKDIM, BLKDIM>>>(d_in, n);
    cudaCheckError();

    /* Copy the result back to host memory */
    cudaSafeCall( cudaMemcpy(in, d_in, SIZE, cudaMemcpyDeviceToHost) );

    /* Free memory on the device */
    cudaFree(d_in);
#endif
}

void fill( int *x, int n )
{
    for (int i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    for (int i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            fprintf(stderr, "Test FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return 0;
        }
    }
    printf("Test OK\n");
    return 1;
}

int main( int argc, char* argv[] )
{
    int *in, *out;
    int n = 1024*1024;
    const int MAX_N = 512*1024*1024;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( n > MAX_N ) {
        fprintf(stderr, "FATAL: input too large (maximum allowed length is %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*in);

    /* Allocate in[] and out[] */
    in = (int*)malloc(SIZE);
    assert(in != NULL);
    out = (int*)malloc(SIZE);
    assert(out != NULL);
    fill(in, n);

    printf("Reverse %d elements... ", n);
    reverse(in, out, n);
    check(out, n);

    printf("In-place reverse %d elements... ", n);
    inplace_reverse(in, n);
    check(in, n);

    /* Cleanup */
    free(in);
    free(out);

    return EXIT_SUCCESS;
}
