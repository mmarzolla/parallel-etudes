/******************************************************************************
 *
 *  PROGRAM: Mandelbrot area
 *
 *  PURPOSE: Program to compute the area of a  Mandelbrot set.
 *           Correct answer should be around 1.510659.
 *           WARNING: this program may contain errors
 *
 *  USAGE:   Program runs without input ... just run the executable
 *
 *  HISTORY: Written:  (Mark Bull, August 2011).
 *           Changed "complex" to "d_complex" to avoid collsion with
 *           math.h complex type (Tim Mattson, September 2011)
 *           Code cleanup (Moreno Marzolla, Feb 2017, Oct 2018, Oct 2020, Jan 2024)
 *
 ******************************************************************************/

/***
% HPC - Area of the Mandelbrot set
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-01-04

The Mandelbrot set is the set of black points in Figure 1.

![Figure 1: The Mandelbrot set](mandelbrot-set.png)

The file [omp-mandelbrot-area.c](omp-mandelbrot-area.c) contains a
serial program that computes an estimate of the area of the Mandelbrot
set.

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

Modify the serial program to use the shared-memory parallelism
provided by OpenMP. To this aim, you can distribute the $N \times N$
lattice points across $P$ OpenMP threads using the `omp parallel for`
directive; you might want to use the `collapse` directive as
well. Each thread computes the number of points that belong to the
Mandelbrot set; the result is the sum-reduction of the partial counts
from each thread. This can be achieved with the `reduction` clause.

Compile with:

        gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot-area.c -o omp-mandelbrot-area

Run with:

        ./omp-mandelbrot-area [N]

For example, to use a grid of $1000 \times 1000$$ points using $P=2$
OpenMP threads:

        OMP_NUM_THREADS=2 ./omp-mandelbrot-area 1000

You might want to experiment with the `static` or `dynamic` scheduling
policies, as well as with some different values for the chunk size.

## Files

- [cuda-mandelbrot-area.c](cuda-mandelbrot-area.c)
- [cuda-mandelbrot-area.cl](cuda-mandelbrot-area.cl)

***/

#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const float XMIN = -2.25f;
const float XMAX = 0.75f;
const float YMIN = -1.4f;
const float YMAX = 1.5f;

#define MAXIT 10000

/**
 * Performs the iteration z = z*z+c, until ||z|| > 2 when point is
 * known to be outside the Mandelbrot set. Return the number of
 * iterations until ||z|| > 2, or MAXIT.
 */
#ifndef SERIAL
__device__
#endif
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

#ifndef SERIAL
#define BLKDIM 32

__global__ void
mandelbrot_area_kernel( int xsize,
                        int ysize,
                        uint32_t *ninside)
{
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uint32_t local_inside;

    if (lx == 0 && ly == 0)
        local_inside = 0;

    __syncthreads();

    if (x < xsize && y < ysize) {
        const float cx = XMIN + (XMAX - XMIN) * x / xsize;
        const float cy = YMIN + (YMAX - YMIN) * y / ysize;
        const int v = iterate(cx, cy);
        if (v >= MAXIT)
            atomicAdd(&local_inside, 1);
    }

    __syncthreads();

    if (lx == 0 && ly == 0)
        atomicAdd(ninside, local_inside);
}
#endif

uint32_t inside( int xsize,int ysize )
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
    const dim3 BLOCK(BLKDIM, BLKDIM);
    const dim3 GRID((xsize + BLKDIM-1) / BLKDIM,
                    (ysize + BLKDIM-1) / BLKDIM);
    uint32_t *d_ninside;
    cudaSafeCall( cudaMalloc((void**)&d_ninside, sizeof(*d_ninside)) );
    cudaSafeCall( cudaMemcpy(d_ninside, &ninside, sizeof(ninside), cudaMemcpyHostToDevice) );
    mandelbrot_area_kernel<<< GRID, BLOCK >>>( xsize, ysize, d_ninside );
    cudaCheckError();
    cudaSafeCall( cudaMemcpy(&ninside, d_ninside, sizeof(ninside), cudaMemcpyDeviceToHost) );
    cudaFree(d_ninside);
#endif
    return ninside;
}

int main( int argc, char *argv[] )
{
    int npoints = 1000;

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
    const double area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/(npoints*npoints);

    printf("Area of Mandlebrot set = %f\n", area);
    printf("Correct answer should be around 1.50659\n");
    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}