/****************************************************************************
 *
 * opencl-edge-detect.c - Edge detection on grayscale images
 *
 * Copyright (C) 2019, 2021, 2022, 2024 Moreno Marzolla
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
% HPC - Edge detection on grayscale images
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-01-04

![Result of the Sobel operator](edge-detect.png)

The [Sobel operator](https://en.wikipedia.org/wiki/Sobel_operator) is
used to detect the edges on an grayscale image. The idea is to compute
the gradient of color change across each pixel; those pixels for which
the gradient exceeds a user-defined threshold are considered to be
part of an edge. Computation of the gradient involves the application
of a $3 \times 3$ stencil to the input image.

The program reads an input image fro standard input in
[PGM](https://en.wikipedia.org/wiki/Netpbm#PGM_example) (_Portable
Graymap_) format and produces a B/W image to standard output. The user
can specify an optional threshold on the command line.

The goal of this exercise is to parallelize the computation of the
Sobel operator using OpenCL; this can be achieved by writing a kernel
that computes the edge at each pixel, and invoke the kernel from the
`edge_detect()` function.

To compile:

        cc opencl-edge-detect.c simpleCL.c -o opencl-edge-detect -lOpenCL

To execute:

        ./opencl-edge-detect [threshold] < input > output

Example:

        ./opencl-edge-detect < BWstop-sign.pgm > BWstop-sign-edges.pgm

## Files

- [opencl-edge-detect.c](opencl-edge-detect.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpe.h)
- [BWstop-sign.pgm](BWstop-sign.pgm)

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
#include "pgmutils.h"

#ifndef SERIAL
sclKernel sobel_kernel;
#endif

/**
 * Edge detection using the Sobel operator
 */
void edge_detect( const PGM_image* in, PGM_image* edges, int threshold )
{
    const int width = in->width;
    const int height = in->height;
#ifdef SERIAL
    /* The following C99 casts are used to convert `in` and `edges` to
       the type "array[width] of pointers to unsigned char", so that
       indexing can be done more easily */
    const unsigned char (*in_bmap)[width] = (const unsigned char (*)[width])(in->bmap);
    unsigned char (*edges_bmap)[width] = (unsigned char (*)[width])(edges->bmap);
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++)  {
            /* Compute the gradients Gx and Gy along the x and y
               dimensions */
            const int Gx =
                in_bmap[i-1][j-1] - in_bmap[i-1][j+1]
                + 2*in_bmap[i][j-1] - 2*in_bmap[i][j+1]
                + in_bmap[i+1][j-1] - in_bmap[i+1][j+1];
            const int Gy =
                in_bmap[i-1][j-1] + 2*in_bmap[i-1][j] + in_bmap[i-1][j+1]
                - in_bmap[i+1][j-1] - 2*in_bmap[i+1][j] - in_bmap[i+1][j+1];
            const int magnitude = Gx * Gx + Gy * Gy;
            if  (magnitude > threshold*threshold)
                edges_bmap[i][j] = WHITE;
            else
                edges_bmap[i][j] = BLACK;
        }
    }
#else
    sclDim block, grid;
    sclWGSetup2D(width, height, &grid, &block);
    const size_t size = width * height;
    cl_mem d_in = sclMallocCopy(size, in->bmap, CL_MEM_READ_ONLY);
    cl_mem d_edges = sclMalloc(size, CL_MEM_WRITE_ONLY);
    sclSetArgsEnqueueKernel(sobel_kernel,
                            grid, block,
                            ":b :b :d :d :d",
                            d_in, d_edges, width, height, threshold);
    sclMemcpyDeviceToHost(edges->bmap, d_edges, size);
    sclFree(d_in);
    sclFree(d_edges);
#endif
}

int main( int argc, char* argv[] )
{
    PGM_image bmap, out;
    int threshold = 70;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [threshold] < in.pgm > out.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }
    if ( argc > 1 ) {
        threshold = atoi(argv[1]);
    }
    read_pgm(stdin, &bmap);
    init_pgm(&out, bmap.width, bmap.height, WHITE);
#ifndef SERIAL
    sclInitFromFile("opencl-edge-detect.cl");
    sobel_kernel = sclCreateKernel("sobel_kernel");
#endif
    const double tstart = hpc_gettime();
    edge_detect(&bmap, &out, threshold);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time %f\n", elapsed);
    write_pgm(stdout, &out, "produced by opencl-edge-detect.c");
    free_pgm(&bmap);
    free_pgm(&out);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
