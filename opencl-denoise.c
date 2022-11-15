/****************************************************************************
 *
 * opencl-denoise.c -- Rimozione del rumore da una immagine
 *
 * Copyright 2018--2022 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Rimozione del rumore da una immagine
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Ultimo aggiornamento 2022-03-21

![By Simpsons contributor, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.
php?curid=8904364>](denoise.png)

Il file [opencl-denoise.c](opencl-denoise.c) contiene una
implementazione seriale di un programma per effettuare il _denoising_,
cioè per rimuovere il "rumore", da una immagine a colori. L'algoritmo
di denoising è molto semplice, e consiste nell'impostare il colore di
ciascun pixel come la mediana dei colori dei quattro pixel adiacenti
più il pixel stesso (_median-of-five_). Questa operazione viene
ripetuta separatamente per ciascuno dei tre canali di colore (rosso,
verde, blu).

Il programma legge l'immagine di input da standard input in formato
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap), e
produce il risultato su standard output nello stesso formato.

Per compilare:

        nvcc opencl-denoise.c -o opencl-denoise

Per eseguire:

        ./opencl-denoise < input > output

Esempio:

        ./opencl-denoise < giornale.ppm > giornale-denoised.ppm

## File

- [opencl-denoise.c](opencl-denoise.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)
- [giornale.ppm](giornale.ppm) (input di esempio)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "simpleCL.h"

#include "ppmutils.h"

#define BLKDIM 32

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

int IDX(int width, int i, int j)
{
    return (i*width + j);
}

/**
 * Return the median of v[0..4]
 */
unsigned char median_of_five( unsigned char v[5] )
{
    /* We do a partial sort of v[5] using bubble sort until v[2] is
       correctly placed; this element is the median. (There are better
       ways to compute the median-of-five). */
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v  , v+1 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    compare_and_swap( v+1, v+2 );
    compare_and_swap( v+3, v+4 );
    compare_and_swap( v+2, v+3 );
    return v[2];
}

/**
 * Denoise a single color channel
 */
#ifdef SERIAL
void denoise( unsigned char *bmap, int width, int height )
{
    unsigned char *out = (unsigned char*)malloc(width*height);
    unsigned char v[5];
    assert(out != NULL);

    memcpy(out, bmap, width*height);
    /* Pay attention to the indexes! */
    for (int i=1; i<height - 1; i++) {
        for (int j=1; j<width - 1; j++) {
            v[0] = bmap[IDX(width, i  , j  )];
            v[1] = bmap[IDX(width, i  , j-1)];
            v[2] = bmap[IDX(width, i  , j+1)];
            v[3] = bmap[IDX(width, i-1, j  )];
            v[4] = bmap[IDX(width, i+1, j  )];

            out[IDX(width, i, j)] = median_of_five(v);
        }
    }
    memcpy(bmap, out, width*height);
    free(out);
}
#else

sclKernel denoise_kernel;

void denoise( unsigned char *bmap, int width, int height )
{
    cl_mem d_bmap, d_out;
    const size_t SIZE = width * height * sizeof(*bmap);
    const sclDim BLOCK = DIM2(SCL_DEFAULT_WG_SIZE2D,
                              SCL_DEFAULT_WG_SIZE2D);
    const sclDim GRID = DIM2(sclRoundUp(width, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(height, SCL_DEFAULT_WG_SIZE2D));

    d_bmap = sclMallocCopy(SIZE, bmap, CL_MEM_READ_ONLY);
    d_out = sclMalloc(SIZE, CL_MEM_WRITE_ONLY);
    sclSetArgsEnqueueKernel(denoise_kernel,
                            GRID, BLOCK,
                            ":b :b :d :d",
                            d_bmap, d_out, width, height);
    sclMemcpyDeviceToHost(bmap, d_out, SIZE);
    sclFree(d_bmap);
    sclFree(d_out);
}
#endif

int main( void )
{
    PPM_image img;
#ifndef SERIAL
    sclInitFromFile("opencl-denoise.cl");
    denoise_kernel = sclCreateKernel("denoise_kernel");
#endif
    read_ppm(stdin, &img);
    const double tstart = hpc_gettime();
    denoise(img.r, img.width, img.height);
    denoise(img.g, img.width, img.height);
    denoise(img.b, img.width, img.height);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time: %f\n", elapsed);
    write_ppm(stdout, &img, "produced by opencl-denoise.c");
    free_ppm(&img);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
