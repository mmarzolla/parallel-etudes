/****************************************************************************
 *
 * mpi-mandelbrot.c - Draw the Mandelbrot set with MPI
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Insieme di Mandelbrot
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento: 2022-06-30

![Benoit Mandelbrot (1924--2010)](Benoit_Mandelbrot.jpg)

Il file [opencl-mandelbrot.c](opencl-mandelbrot.c) contiene lo
scheletro di un programma OpenCL che calcola l'insieme di
Mandelbrot.

Il programma accetta come parametro opzionale la dimensione verticale
dell'immagine, ossia il numero di righe (default 1024). La risoluzione
orizzontale viene calcolata automaticamente dal programma in modo da
includere l'intero insieme. Il programma produce un file
`mandebrot.ppm` contenente una immagine dell'insieme di Mandelbrot in
formato PPM (_Portable Pixmap_). Se non si dispone di un programma per
visualizzare questo formato, lo si può convertire, ad esempio, in PNG
dando sul server il comando:

        convert mandelbrot.ppm mandelbrot.png

Scopo di questo esercizio è quello di sviluppare una versione
parallela usando OpenCL.

[TBD]

Suggerisco di conservare la versione seriale del programma per usarla
come riferimento. Per verificare in modo empirico la correttezza del
programma parallelo, consiglio di confrontare il risultato con quello
prodotto dalla versione seriale: le due immagini devono risultare
identiche byte per byte. Per confrontare due immagini si può usare il
comando `cmp` dalla shell di Linux:

        cmp file1 file2

stampa un messaggio se e solo se `file1` e `file2` differiscono.

Per compilare:

        cc -std=c99 -Wall -Wpedantic opencl-mandelbrot.c simpleCL.c -o opencl-mandelbrot

Per eseguire:

        ./opencl-mandelbrot [ysize]

scrive il risultato sul file `opencl-mandelbrot.ppm`

## File

- [opencl-mandelbrot.c](opencl-mandelbrot.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "simpleCL.h"

typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

int main( int argc, char *argv[] )
{
    FILE *out = NULL;
    const char* fname="opencl-mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    cl_mem d_bitmap;
    int xsize, ysize;
    sclKernel mandelbrot_kernel;

    sclInitFromFile("opencl-mandelbrot.cl");
    mandelbrot_kernel = sclCreateKernel("mandelbrot_kernel");

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }

    xsize = ysize * 1.4;

    out = fopen(fname, "w");
    if ( !out ) {
        fprintf(stderr, "Error: cannot create %s\n", fname);
        return EXIT_FAILURE;
    }

    /* Write the header of the output file */
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", xsize, ysize);
    fprintf(out, "255\n");

    const size_t BMAP_SIZE = xsize * ysize * sizeof(pixel_t);

    bitmap = (pixel_t*)malloc(BMAP_SIZE); assert(bitmap != NULL);
    d_bitmap = sclMalloc(BMAP_SIZE, CL_MEM_WRITE_ONLY);

    const sclDim BLOCK = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim GRID = DIM2(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(ysize, SCL_DEFAULT_WG_SIZE2D));

    const double tstart = hpc_gettime();

    sclSetArgsEnqueueKernel(mandelbrot_kernel,
                            GRID, BLOCK,
                            ":d :d :b",
                            xsize, ysize, d_bitmap);

    sclMemcpyDeviceToHost(bitmap, d_bitmap, BMAP_SIZE);

    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
    fclose(out);

    free(bitmap);
    sclFree(d_bitmap);

    sclFinalize();
    
    return EXIT_SUCCESS;
}
