/****************************************************************************
 *
 * cuda-mandelbrot.c - Mandelbrot set
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Mandelbrot set
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-09-03

![](mandelbrot-set.png)

The file [cuda-mandelbrot.cu](cuda-mandelbrot.cu) contains a serial
program that computes the Mandelbrot set. The program accepts the
image height as an optional command-line parameter; the width is
computed automatically to include the whole set. The program writes a
graphical representation of the Mandelbrot set into a file
`mandebrot.ppm` in PPM (_Portable Pixmap_) format. If you don't have a
suitable viewer, you can convert the image, e.g., into PNG with the
command:

        convert mandelbrot.ppm mandelbrot.png

The goal of this exercise is to write a parallel version of the
program using CUDA. A 2D grid of 2D blocks is created, and each thread
takes care of computing a single pixel of the image.  The size of each
thread block is $\mathit{BLKDIM} \times \mathit{BLKDIM}$. The side of
the grid is the minimum integer multiple of _BLKDIM_ that covers the
whole image:

```C
const dim3 BLOCK(BLKDIM, BLKDIM);
const dim3 GRID((xsize + BLKDIM-1)/BLKDIM, (ysize + BLKDIM-1)/BLKDIM);
```

You may want to keep the serial program as a reference; to check the
correctness of the parallel implementation, you can compare the output
images produced by both versions with the command:

        cmp file1 file2

Both images should be identical; if not, something is wrong.

To compile:

        nvcc cuda-mandelbrot.c -o cuda-mandelbrot

To execute:

        ./cuda-mandelbrot [ysize]

Example:

        ./cuda-mandelbrot 800

## Files

- [cuda-mandelbrot.cu](cuda-mandelbrot.cu)

***/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "hpc.h"

#define BLKDIM 32
#define MAXIT 100

typedef struct __attribute__((__packed__)) {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/* color gradient from https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia */
__constant__ pixel_t colors[] = {
    {66, 30, 15}, /* r, g, b */
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3} };
__constant__ int NCOLORS = sizeof(colors)/sizeof(colors[0]);

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first `n` such that `z_n > bound`, or `MAXIT` if `z_n` is below
 * `bound` after `MAXIT` iterations.
 */
__device__
int iterate( float cx, float cy )
{
    float x = 0.0f, y = 0.0f, xnew, ynew;
    int it;
    for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0f*2.0f); it++ ) {
        xnew = x*x - y*y + cx;
        ynew = 2.0f*x*y + cy;
        x = xnew;
        y = ynew;
    }
    return it;
}

/* Draw the rows of the Mandelbrot set from `ystart` (inclusive) to
   `yend` (excluded) to the bitmap pointed to by `p`. Note that `p`
   must point to the beginning of the bitmap where the portion of
   image will be stored; in other words, this function writes to
   pixels p[0], p[1], ... `xsize` and `ysize` MUST be the sizes
   of the WHOLE image. */
__global__ void
mandelbrot_kernel( int xsize, int ysize, pixel_t* bmap)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < xsize && y < ysize) {
        pixel_t *p = bmap + y * xsize + x;
        const float cx = -2.5f + 3.5f * (float)x / (xsize - 1);
        const float cy = 1 - 2.0f * (float)y / (ysize - 1);
        const int v = iterate(cx, cy);
        if (v < MAXIT) {
            p->r = colors[v % NCOLORS].r;
            p->g = colors[v % NCOLORS].g;
            p->b = colors[v % NCOLORS].b;
        } else {
            p->r = p->g = p->b = 0;
        }
    }
}

int main( int argc, char *argv[] )
{
    FILE *out = NULL;
    const char* fname="cuda-mandelbrot.ppm";
    pixel_t *bitmap = NULL, *d_bitmap;
    int xsize, ysize;

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
    cudaSafeCall( cudaMalloc((void**)&d_bitmap, BMAP_SIZE) );

    const dim3 BLOCK(BLKDIM, BLKDIM);
    const dim3 GRID((xsize + BLKDIM-1)/BLKDIM, (ysize + BLKDIM-1)/BLKDIM);

    const double tstart = hpc_gettime();
    mandelbrot_kernel<<<GRID, BLOCK>>>(xsize, ysize, d_bitmap);

    cudaSafeCall( cudaMemcpy(bitmap, d_bitmap, BMAP_SIZE, cudaMemcpyDeviceToHost) );

    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Elapsed time (s) : %f\n", elapsed);

    fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
    fclose(out);

    free(bitmap);
    cudaSafeCall( cudaFree(d_bitmap) );

    return EXIT_SUCCESS;
}