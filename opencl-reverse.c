/****************************************************************************
 *
 * opencl-reverse.c - Array reversal with OpenCL
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
% Ultimo aggiornamento: 2021-12-03

Realizzare un programma per invertire un array di $n$ elementi interi,
cioè scambiare il primo elemento con l'ultimo, il secondo col
penultimo e così via. Realizzare due kernel diversi: il primo ricopia
gli elementi di un array `in[]` in un altro array `out[]` in modo che
quest'ultimo contenga gli stessi elementi di in `in[]` in ordine
inverso; il secondo kernel inverte gli elementi di `in[]` "in place",
ossia modificando `in[]` senza sfruttare altri array di appoggio.

Il file [opencl-reverse.c](opencl-reverse.c) fornisce una
implementazione basata su CPU delle funzioni `reverse()` e
`inplace_reverse()`; modificare il programma per trasformare le
funzioni in kernel da invocare opportunamente.

**Suggerimento:** la funzione `reverse()` può essere facilmente
trasformata in un kernel eseguito da $n$ work-item (uno per ogni
elemento dell'array). Ciascun work-item copia un elemento di `in[]`
nella corretta posizione di `out[]`; utilizzare workgroup 1D, dato che
in tal caso risulta facile mappare ciascun work-item in un elemento
dell'array di input. La funzione `inplace_reverse()` si trasforma in
un kernel in modo simile, che però verrà eseguito da $n/2$ work-item
anziché $n$; ciascuno degli $n/2$ work-item scambia un elemento della
prima metà dell'array `in[]` con l'elemento in posizione simmetrica
nella seconda metà. Controllare che il programma funzioni anche se $n$
è dispari.

Per compilare:

        cc opencl-reverse.c simpleCL.c -o opencl-reverse -lOpenCL

Per eseguire:

        ./opencl-reverse [n]

Esempio:

        ./opencl-reverse

## File

- [opencl-reverse.c](opencl-reverse.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "simpleCL.h"

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
    cl_mem d_in, d_out; /* device copy of in and out */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] and out[] */
    d_in = sclMallocCopy(SIZE, in, CL_MEM_READ_ONLY);
    d_out = sclMalloc(SIZE, CL_MEM_WRITE_ONLY);

    /* Launch the reverse() kernel on the GPU */
    sclSetArgsLaunchKernel(sclCreateKernel("reverse_kernel"),
                           DIM1(sclRoundUp(n, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :b :d",
                           d_in, d_out, n);

    /* Copy the result back to host memory */
    sclMemcpyDeviceToHost(out, d_out, SIZE);

    /* Free memory on the device */
    sclFree(d_in);
    sclFree(d_out);
#endif
}

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
    cl_mem d_in; /* device copy of in */
    const size_t SIZE = n * sizeof(*in);

    /* Allocate device copy of in[] */
    d_in = sclMallocCopy(SIZE, in, CL_MEM_READ_WRITE);

    /* Launch the reverse() kernel on the GPU */
    sclSetArgsLaunchKernel(sclCreateKernel("inplace_reverse_kernel"),
                           DIM1(sclRoundUp(n/2, SCL_DEFAULT_WG_SIZE)), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :d",
                           d_in, n);

    /* Copy the result back to host memory */
    sclMemcpyDeviceToHost(in, d_in, SIZE);

    /* Free memory on the device */
    sclFree(d_in);
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
#ifndef SERIAL
    sclInitFromFile("opencl-reverse.cl");
#endif
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
    in = (int*)malloc(SIZE); assert(in != NULL);
    out = (int*)malloc(SIZE); assert(out != NULL);
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

#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
