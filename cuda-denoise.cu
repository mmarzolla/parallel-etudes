/****************************************************************************
 *
 * cuda-denoise.cu -- Rimozione del rumore da una immagine
 *
 * Copyright 2018--2021 Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Ultimo aggiornamento 2021-11-26

Il file [cuda-denoise.cu](cuda-denoise.cu) contiene una
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

        nvcc cuda-denoise.cu -o cuda-denoise

Per eseguire:

        ./cuda-denoise < input > output

Esempio:

        ./cuda-denoise < giornale.ppm > giornale-denoised.ppm

## File

- [cuda-denoise.cu](cuda-denoise.cu)
- [giornale.ppm](giornale.ppm) (input di esempio)

 ***/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "ppmutils.h"

#define BLKDIM 32

/**
 * Swap *a and *b if necessary so that, at the end, *a <= *b
 */
#ifndef SERIAL
__device__
#endif
void compare_and_swap( unsigned char *a, unsigned char *b )
{
    if (*a > *b ) {
        unsigned char tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

#ifndef SERIAL
__device__
#endif
unsigned char *PTR(unsigned char *bmap, int width, int i, int j)
{
    return (bmap + i*width + j);
}

/**
 * Return the median of v[0..4]
 */
#ifndef SERIAL
__device__
#endif
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
            v[0] = *PTR(bmap, width, i  , j  );
            v[1] = *PTR(bmap, width, i  , j-1);
            v[2] = *PTR(bmap, width, i  , j+1);
            v[3] = *PTR(bmap, width, i-1, j  );
            v[4] = *PTR(bmap, width, i+1, j  );

            *PTR(out, width, i, j) = median_of_five(v);
        }
    }
    memcpy(bmap, out, width*height);
    free(out);
}
#else
__global__ void denoise_kernel( unsigned char *bmap, unsigned char *out, int width, int height )
{
    const int i = threadIdx.y + blockIdx.y * blockDim.y;
    const int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i<height && j<width) {
        if ((i>0) && (i<height-1) && (j>0) && (j<width-1)) {
            unsigned char v[5];
            v[0] = *PTR(bmap, width, i  , j  );
            v[1] = *PTR(bmap, width, i  , j-1);
            v[2] = *PTR(bmap, width, i  , j+1);
            v[3] = *PTR(bmap, width, i-1, j  );
            v[4] = *PTR(bmap, width, i+1, j  );

            *PTR(out, width, i, j) = median_of_five(v);
        } else {
            *PTR(out, width, i, j) = *PTR(bmap, width, i, j);
        }
    }
}

void denoise( unsigned char *bmap, int width, int height )
{
    unsigned char *d_bmap, *d_out;
    const size_t SIZE = width * height * sizeof(*bmap);
    const dim3 BLOCK(BLKDIM, BLKDIM);
    const dim3 GRID((width + BLKDIM-1)/BLKDIM, (height + BLKDIM-1)/BLKDIM);

    cudaSafeCall(cudaMalloc((void**)&d_bmap, SIZE));
    cudaSafeCall(cudaMalloc((void**)&d_out, SIZE));
    cudaSafeCall(cudaMemcpy(d_bmap, bmap, SIZE, cudaMemcpyHostToDevice));
    denoise_kernel<<<GRID, BLOCK>>>(d_bmap, d_out, width, height); cudaCheckError();
    cudaSafeCall(cudaMemcpy(bmap, d_out, SIZE, cudaMemcpyDeviceToHost));
    cudaFree(d_bmap);
    cudaFree(d_out);
}
#endif

int main( void )
{
    PPM_image img;
    read_ppm(stdin, &img);
    const double tstart = hpc_gettime();
    denoise(img.r, img.width, img.height);
    denoise(img.g, img.width, img.height);
    denoise(img.b, img.width, img.height);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time: %f\n", elapsed);
    write_ppm(stdout, &img, "produced by cuda-denoise.cu");
    free_ppm(&img);
    return EXIT_SUCCESS;
}
