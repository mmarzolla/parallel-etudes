/****************************************************************************
 *
 * opencl-denoise.c - Image denoising using the median filter
 *
 * Copyright (C) 2018--2024 Moreno Marzolla
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
% Image denoising using the median filter
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-01-04

![By Simpsons contributor, CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.
php?curid=8904364>](denoise.png)

The file [opencl-denoise.c](opencl-denoise.c) contains a serial
implementaiton of a program for _image denoising_, that is, to remove
"noise" from a color image. The algorithm is based on a _median
filter_: the color of each pixel is computed as the median of the
colors of the four adjacent pixels, plus itself
(_median-of-five_). This operation is applied separately for each
color channel (red, green, blue).

The program reads the input image from standard input in
[PPM](http://netpbm.sourceforge.net/doc/ppm.html) (Portable Pixmap)
format, and outputs the result to standard output.

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-denoise.c simpleCL.c -o opencl.denoise -lOpenCL

To execute:

        ./opencl-denoise < input > output

Example:

        ./opencl-denoise < valve-noise.ppm > valve-denoised.ppm

## File

- [opencl-denoise.c](opencl-denoise.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)
- [valve-noise.ppm](valve-noise.ppm) (sample input)

 ***/

/* The following #define is required by the implementation of
   hpc_gettime(). It MUST be defined before including any other
   file. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
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
