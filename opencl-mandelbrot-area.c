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
 *           Code cleanup (Moreno Marzolla, Feb 2017, Oct 2018, Oct 2020)
 *
 ******************************************************************************/

/***
% HPC - Area of the Mandelbrot set
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-08-16

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
Mandelbrot set; the result is simply the sum-reduction of the partial
counts from each thread. This can be achieved with the `reduction`
clause.

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

- [opencl-mandelbrot-area.c](opencl-mandelbrot-area.c)
- [opencl-mandelbrot-area.cl](opencl-mandelbrot-area.cl)

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include "simpleCL.h"

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const double XMIN = -2.25, XMAX = 0.75;
const double YMIN = -1.5, YMAX = 1.5;

int main( int argc, char *argv[] )
{
    uint32_t ninside = 0;
    int npoints = 1000;
    sclKernel mandelbrot_area_kernel;
    cl_mem d_ninside;

    sclInitFromFile("opencl-mandelbrot-area.cl");
    mandelbrot_area_kernel = sclCreateKernel("mandelbrot_area_kernel");

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [npoints]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        npoints = atoi(argv[1]);
    }

    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE1D);
    const sclDim GRID = DIM1(sclRoundUp(npoints, SCL_DEFAULT_WG_SIZE1D));
    d_ninside = sclMallocCopy(sizeof(ninside), &ninside, CL_MEM_READ_WRITE);

    const double tstart = hpc_gettime();
    sclSetArgsEnqueueKernel(mandelbrot_area_kernel,
                            GRID, BLOCK,
                            ":d :d :b",
                            12345u, npoints, d_ninside);
    sclDeviceSynchronize();
    const double elapsed = hpc_gettime() - tstart;
    sclMemcpyDeviceToHost(&ninside, d_ninside, sizeof(ninside));
    printf("npoints = %d, ninside = %u\n", npoints, ninside);

    /* Compute area and error estimate and output the results */
    const double area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/npoints;

    printf("Area of Mandlebrot set = %12.8f\n", area);
    printf("Correct answer should be around 1.50659\n");
    printf("Elapsed time: %f\n", elapsed);
    sclFree(d_ninside);
    sclFinalize();
    return EXIT_SUCCESS;
}
