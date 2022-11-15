/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
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
% HPC - Inversione di un array
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-10-15

Realizzare un programma per invertire un array di $n$ elementi interi,
cioè scambiare il primo elemento con l'ultimo, il secondo col
penultimo e così via. Realizzare due kernel diversi: il primo ricopia
gli elementi di un array `in[]` in un altro array `out[]` in modo che
quest'ultimo contenga gli stessi elementi di in `in[]` in ordine
inverso; il secondo kernel inverte gli elementi di `in[]` "in place",
ossia modificando `in[]` senza sfruttare altri array di appoggio.

Il file [cuda-reverse.cu](cuda-reverse.cu) fornisce una
implementazione basata su CPU delle funzioni `reverse()` e
`inplace_reverse()`; modificare il programma per trasformare le
funzioni in kernel da invocare opportunamente.

**Suggerimento:** la funzione `reverse()` può essere facilmente
trasformata in un kernel eseguito da $n$ CUDA thread (uno per ogni
elemento dell'array). Ciascun thread copia un elemento di `in[]` nella
corretta posizione di `out[]`; utilizzare _thread block_ 1D, dato che
in tal caso risulta facile mappare ciascun thread in un elemento
dell'array di input. La funzione `inplace_reverse()` si trasforma in
un kernel in modo simile, che però verrà eseguito da $n/2$ CUDA thread
anziché $n$; ciascuno degli $n/2$ thread scambia un elemento della
prima metà dell'array `in[]` con l'elemento in posizione simmetrica
nella seconda metà. Controllare che il programma funzioni anche se $n$
è dispari.

Per compilare:

        nvcc cuda-reverse.cu -o cuda-reverse

Per eseguire:

        ./cuda-reverse [n]

Esempio:

        ./cuda-reverse

## File

- [cuda-reverse.cu](cuda-reverse.cu)
- [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef SERIAL
#define BLKDIM 1024

/* Reverse in[] into out[]; n CUDA threads are required to reverse n
   elements */
__global__ void reverse_kernel( int *in, int *out, int n )
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n ) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
}
#endif

/* Reverse in[] into out[].

   [TODO] Modify this function so that it:
   - allocates memory on the device to hold a copy of `in` and `out`;
   - copies `in` and `out` to the device
   - launches a kernel (to be defined)
   - copies data back from device to host
   - deallocates memory on the device
 */
void reverse( int *in, int *out, int n )
{
#ifdef SERIAL
    int i;
    for (i=0; i<n; i++) {
        const int opp = n - 1 - i;
        out[opp] = in[i];
    }
#else
    int *d_in, *d_out; /* device copy of in and out */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] and out[] */
    cudaSafeCall( cudaMalloc((void **)&d_in, SIZE) );
    cudaSafeCall( cudaMalloc((void **)&d_out, SIZE) );

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );

    /* Launch the reverse() kernel on the GPU */
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
/* In-place reversal of in[]; n/2 CUDA threads are required to reverse
   n elements */
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

    /* Allocate device copy of in[] */
    cudaSafeCall( cudaMalloc((void **)&d_in, SIZE) );

    /* Copy input to device */
    cudaSafeCall( cudaMemcpy(d_in, in, SIZE, cudaMemcpyHostToDevice) );

    /* Launch the reverse() kernel on the GPU */
    inplace_reverse_kernel<<<(n/2 + BLKDIM-1)/BLKDIM, BLKDIM>>>(d_in, n);
    cudaCheckError();

    /* Copy the result back to host memory */
    cudaSafeCall( cudaMemcpy(in, d_in, SIZE, cudaMemcpyDeviceToHost) );

    /* Free memory on the device */
    cudaFree(d_in);
#endif
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( const int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
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
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
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
