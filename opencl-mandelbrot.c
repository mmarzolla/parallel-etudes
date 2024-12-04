/****************************************************************************
 *
 * opencl-mandelbrot.c - Mandelbrot set
 *
 * Copyright (C) 2017--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Mandelbrot set
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

![](mandelbrot-set.png)

The file [opencl-mandelbrot.c](opencl-mandelbrot.c) contains a serial
program that computes the Mandelbrot set. The program accepts the
image height as an optional command-line parameter; the width is
computed automatically to include the whole set. The program writes a
graphical representation of the Mandelbrot set into a file
`mandebrot.ppm` in PPM (_Portable Pixmap_) format. If you don't have a
suitable viewer, you can convert the image, e.g., into PNG with the
command:

        convert mandelbrot.ppm mandelbrot.png

The goal of this exercise is to write a parallel version of the
program using OpenCL. A 2D grid of 2D workgroups is created, and each
work-item takes care of computing a single pixel of the image.  The
size of each workgroup is $\mathit{SCL\_DEFAULT\_WG\_SIZE2D} \times
\mathit{SCL\_DEFAULT\_WG\_SIZE2D}$. The side of the grid is the
minimum integer multiple of _SCL_DEFAULT_WG_SIZE2D_ that covers the
whole image. The `simpleCL` library provides a function `sclRoundUp(n,
a)` that can be used to compute the minimum integer multiple of _a_
that is no less than _n_; therefore, the size of a workgroup and grid
can be computed as:

```C
const sclDim BLOCK = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
const sclDim GRID = DIM2(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE2D),
                         sclRoundUp(ysize, SCL_DEFAULT_WG_SIZE2D));
```

You may want to keep the serial program as a reference; to check the
correctness of the parallel implementation, you can compare the output
images produced by both versions with the command:

        cmp file1 file2

Both images should be identical; if not, something is wrong.

To compile:

        cc -std=c99 -Wall -Wpedantic opencl-mandelbrot.c simpleCL.c -o opencl-mandelbrot -lOpenCL

To execute:

        ./opencl-mandelbrot [ysize]

Example:

        ./opencl-mandelbrot 800

## Files

- [opencl-mandelbrot.c](opencl-mandelbrot.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h)

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
