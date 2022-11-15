/****************************************************************************
 *
 * opencl-dot.c - Dot product
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
% HPC - Prodotto scalare
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2021-12-04

## Familiarizzare con l'ambiente di lavoro

Il server `isi-raptor03.csr.unibo.it` dispone di tre GPU identiche
(NVidia GeForce GTX 1070). Di default viene utilizzata la prima; è
però possibile selezionare il dispositivo OpenCL da usare mediante la
variabile d'ambiente `SCL_DEFAULT_DEVICE`; ad esempio

        SCL_DEFAULT_DEVUCE=1 ./opencl-dot

esegue il programma `opencl-dot` sulla seconda GPU.

Il comando `clinfo` visualizza le caratteristiche dei dispotivi OpenCL
disponibili (tra i quali figura anche la CPU).

## Prodotto scalare

Modificare il file [opencl-dot.c](opencl-dot.c) per calcolare e
stampare il prodotto scalare tra due array `x[]` e `y[]` di lunghezza
$n$ sfruttando OpenCL, trasformando la funzione `dot()` in un
kernel. Ricordiamo che il prodotto scalare $s$ di due array `x[]` e
`y[]` è definito come

$$
s = \sum_{i=0}^{n-1} x[i] \times y[i]
$$

Sono necessarie delle modifiche alla funzione `dot()` per sfruttare la
GPU. Per questo esercizio si richiede l'uso di un singolo workgroup
composto da `SCL_DEFAULT_WG_SIZE` work-item, procedendo come segue:

1. La CPU alloca sul device un array `tmp[]` di `SCL_DEFAULT_WG_SIZE`
   elementi, oltre ad una copia degli array `x[]` e `y[]`. Per
   allocare gli array si usino le funzioni `sclMallocCopy()`
   e `sclMalloc()`.

2. La CPU esegue il kernel che calcola il prodotto scalare come segue:
   il work-item $t$ calcola il valore dell'espressione $(x[t] \times
   y[t] + x[t + B] \times y[t + B] + x[t + 2 \times B] \times y[t +
   2B] + \ldots$) e memorizza il risultato in `tmp[t]`.

3. Una volta che il kernel termina l'esecuzione, la CPU trasferisce
   l'array `tmp[]` dalla memoria del device a quella dell'host, e ne
   somma il contenuto determinando così il prodotto scalare cercato.

Si rende quindi necessario calcolare il prodotto scalare in due fasi:
la prima (passo 2) viene svolta dal device, mentre la seconda (passo
3) viene svolta dalla CPU. La Figura 1 mostra l'assegnazione del
calcolo dei prodotti scalari ai work-item, assumendo valori "piccoli"
della dimensione del workgroup per semplificare la figura:

![Figura 1](opencl-dot.png)

Il programma deve funzionare correttamente per qualunque valore di $n$

Compilare con:

        cc opencl-dot.c simpleCL.c -o opencl-dot -lm -lOpenCL

Eseguire con:

        ./opencl-dot [len]

Esempio:

        ./opencl-dot

## File

- [opencl-dot.c](opencl-dot.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "simpleCL.h"

#ifndef SERIAL
const char *program =
"__kernel void dot_kernel( __global const float *x,\n"
"                          __global const float *y,\n"
"                          int n,\n"
"                          __global float *tmp )\n"
"{\n"
"    const int tid = get_local_id(0);\n"
"    const int local_size = get_local_size(0);\n"
"    int i;\n"
"    float s = 0.0;\n"
"    for (i = tid; i < n; i += local_size) {\n"
"        s += x[i] * y[i];\n"
"    }\n"
"    tmp[tid] = s;\n"
"}\n";

sclKernel dot_kernel;
#endif

float dot( float *x, float *y, int n )
{
#ifdef SERIAL
    /* [TODO] modify this function so that (part of) the dot product
       computation is executed on the GPU. */
    float result = 0.0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
#else
    float tmp[SCL_DEFAULT_WG_SIZE];
    cl_mem d_x, d_y, d_tmp; /* device copies of x, y, tmp */
    const size_t SIZE_TMP = sizeof(tmp);
    const size_t SIZE_XY = n*sizeof(*x);

    /* Allocate space for device copies of x, y */
    d_x = sclMallocCopy(SIZE_XY, x, CL_MEM_READ_ONLY);
    d_y = sclMallocCopy(SIZE_XY, y, CL_MEM_READ_ONLY);
    d_tmp = sclMalloc(SIZE_TMP, CL_MEM_WRITE_ONLY);

    /* Launch dot_kernel() on the GPU */
    sclSetArgsLaunchKernel(dot_kernel,
                           DIM1(SCL_DEFAULT_WG_SIZE), DIM1(SCL_DEFAULT_WG_SIZE),
                           ":b :b :d :b",
                           d_x, d_y, n, d_tmp);

    /* Copy result back to host */
    sclMemcpyDeviceToHost(tmp, d_tmp, SIZE_TMP);

    /* Perform the last reduction on the CPU */
    float result = 0.0;
    for (int i=0; i<SCL_DEFAULT_WG_SIZE; i++) {
        result += tmp[i];
    }

    /* Cleanup */
    sclFree(d_x);
    sclFree(d_y);
    sclFree(d_tmp);

    return result;
#endif
}

void vec_init( float *x, float *y, int n )
{
    int i;
    const float tx[] = {1.0/64.0, 1.0/128.0, 1.0/256.0};
    const float ty[] = {1.0, 2.0, 4.0};
    const size_t LEN = sizeof(tx)/sizeof(tx[0]);

    for (i=0; i<n; i++) {
        x[i] = tx[i % LEN];
        y[i] = ty[i % LEN];
    }
}

int main( int argc, char* argv[] )
{
    float *x, *y, result;
    int n = 1024*1024;
    const int MAX_N = 128 * n;

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

#ifndef SERIAL
    sclInitFromString(program);
    dot_kernel = sclCreateKernel("dot_kernel");
#endif
    const size_t SIZE = n*sizeof(*x);

    /* Allocate space for host copies of x, y */
    x = (float*)malloc(SIZE);
    assert(x != NULL);
    y = (float*)malloc(SIZE);
    assert(y != NULL);
    vec_init(x, y, n);

    printf("Computing the dot product of %d elements... ", n);
    result = dot(x, y, n);
    printf("result=%f\n", result);

    const float expected = ((float)n)/64;

    /* Check result */
    if ( fabs(result - expected) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED: got %f, expected %f\n", result, expected);
    }

    /* Cleanup */
    free(x);
    free(y);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
