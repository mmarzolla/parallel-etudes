/****************************************************************************
 *
 * cuda-denoise.cu - Image denoising
 *
 * Copyright (C) 2018--2025 Moreno Marzolla
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
% HPC - Image denoising
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-09

![Figure 1: Denoising example (original image by Simpsons, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=8904364>).](denoise.png)

The file [cuda-denoise.c](cuda-denoise.c) contains a serial
implementation of an _image denoising_ algorithm that (to some extent)
can be used to "cleanup" color images. The algorithm replaces the
color of each pixel with the _median_ of the four adjacent pixels plus
itself (_median-of-five_).  The median-of-five algorithm is applied
separately for each color channel (red, green, and blue).

This is particularly useful for removing "hot pixels", i.e., pixels
whose color is way off its intended value, for example due to problems
in the sensor used to acquire the image. However, depending on the
amount of noise, a single pass could be insufficient to remove every
hot pixel; see Figure 1.

The goal of this exercise is to parallelize the denoising algorithm on
the GPU using CUDA. You should launch as many CUDA threads as pixels
in the image, so that each thread is mapped onto a different pixel.

The input image is read from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format; the result is written to standard output in the same format.

To compile:

        nvcc cuda-denoise.cu -o cuda-denoise

To execute:

        ./cuda-denoise < input > output

Example:

        ./cuda-denoise < valve-noise.ppm > valve-denoised.ppm

## Files

- [cuda-denoise.cu](cuda-denoise.cu) [hpc.h](hpc.h)
- [valve-noise.ppm](valve-noise.ppm) (sample input)

 ***/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "hpc.h"
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
    /* Note that the pixels on the border are left unchanged */
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
        /* Note that the pixels on the border are left unchanged */
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
    fprintf(stderr, "Execution time %.3f\n", elapsed);
    write_ppm(stdout, &img, "produced by cuda-denoise.cu");
    free_ppm(&img);
    return EXIT_SUCCESS;
}
