/****************************************************************************
 *
 * opencl-mandelbrot-area.c - Area of the Mandelbrot set
 *
 * Copyright (C) 2022--2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Area of the Mandelbrot set
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

The Mandelbrot set is the set of black points in Figure 1.

![Figure 1: The Mandelbrot set](mandelbrot-set.png)

The file [opencl-mandelbrot-area.c](opencl-mandelbrot-area.c) contains
a serial program that computes an estimate of the area of the
Mandelbrot set.

The program works as follows. First, we identify a rectangle in the
complex plane that contains the Mandelbrot set. Let _(XMIN, YMIN)_ and
_(XMAX, YMAX)_ be the coordinates of the opposite vertices of such a
rectangle (the program contains suitable constants for these
parameters). The bounding rectangle does not need to be tight.

The program overlaps a regular grid of $N \times N$ points over the
bounding rectangle. For each point we decide whether it belongs to the
Mandelbrot set. Let $x$ be the number of random points that do belong
to the Mandelbrot set (by construction, $x \leq N \times N$). Let $B$
be the area of the bounding rectangle defined as

$$
B := (\mathrm{XMAX} - \mathrm{XMIN}) \times (\mathrm{YMAX} - \mathrm{YMIN})
$$

Then, the area $A$ of the Mandelbrot set can be approximated as

$$
A \approx \frac{x}{N^2} \times B
$$

The approximation gets better if the number of points $N$ is large.
The exact value of $A$ is not known, but there are [some
estimates](https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html).

Modify the serial program to use OpenCL parallelism.

Compile with:

        gcc -std=c99 -Wall -Wpedantic opencl-mandelbrot-area.c simpleCL.c -o opencl-mandelbrot-area

Run with:

        ./opencl-mandelbrot-area [N]

For example, to use a grid of $1000 \times 1000$$ points:

        ./opencl-mandelbrot-area 1000

## Files

- [opencl-mandelbrot-area.c](opencl-mandelbrot-area.c) [opencl-mandelbrot-area.cl](opencl-mandelbrot-area.cl)
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
#include "simpleCL.h"

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const double XMIN = -2.25, XMAX = 0.75;
const double YMIN = -1.4, YMAX = 1.5;

const int MAXIT = 10000;

/**
 * Performs the iteration z = z*z+c, until ||z|| > 2 when point is
 * known to be outside the Mandelbrot set. Return the number of
 * iterations until ||z|| > 2, or MAXIT.
 */
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

uint32_t inside( int xsize, int ysize )
{
    uint32_t ninside = 0;
#ifdef SERIAL
    for (int i=0; i<ysize; i++) {
        for (int j=0; j<xsize; j++) {
            const float cx = XMIN + (XMAX-XMIN)*j/xsize;
            const float cy = YMIN + (YMAX-YMIN)*i/ysize;
            ninside += inside(cx, cy);
        }
    }
#else
    cl_mem d_ninside;
    const sclDim BLOCK = DIM2(SCL_DEFAULT_WG_SIZE2D, SCL_DEFAULT_WG_SIZE2D);
    const sclDim GRID = DIM2(sclRoundUp(xsize, SCL_DEFAULT_WG_SIZE2D),
                             sclRoundUp(ysize, SCL_DEFAULT_WG_SIZE2D));
    d_ninside = sclMallocCopy(sizeof(ninside), &ninside, CL_MEM_READ_WRITE);

    sclKernel mandelbrot_area_kernel = sclCreateKernel("mandelbrot_area_kernel");
    sclSetArgsEnqueueKernel(mandelbrot_area_kernel,
                            GRID, BLOCK,
                            ":d :d :b",
                            xsize, ysize, d_ninside);
    sclMemcpyDeviceToHost(&ninside, d_ninside, sizeof(ninside));
    sclFree(d_ninside);
#endif
    return ninside;
}

int main( int argc, char *argv[] )
{
    int npoints = 1000;
#ifndef SERIAL
    sclInitFromFile("opencl-mandelbrot-area.cl");
#endif
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [npoints]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        npoints = atoi(argv[1]);
    }

    printf("Using a %d x %d grid\n", npoints, npoints);

    const double tstart = hpc_gettime();
    const uint32_t ninside = inside(npoints, npoints);
    const double elapsed = hpc_gettime() - tstart;
    printf("npoints = %d, ninside = %u\n", npoints*npoints, ninside);

    /* Compute area and error estimate and output the results */
    const float area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/(npoints*npoints);

    printf("Area of Mandlebrot set = %f\n", area);
    printf("Correct answer should be around 1.50659\n");
    printf("Elapsed time: %f\n", elapsed);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
