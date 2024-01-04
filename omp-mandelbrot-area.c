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
% Last updated: 2022-08-17

The Mandelbrot set is the set of black points in Figure 1.

![Figure 1: The Mandelbrot set](mandelbrot-set.png)

The file [omp-mandelbrot-area.c](omp-mandelbrot-area.c) contains a
serial program that computes an estimate of the area of the Mandelbrot
set.

The program works as follows. First, we identify a rectangle in the
complex plane that contains the Mandelbrot set. Let _(XMIN, YMIN)_ and
_(XMAX, YMAX)_ be the upper left and lower right coordinates of such a
rectangle (the program defines these values).

The program overlaps a regular grid of $N \times N$ points over the
bounding rectangle. For each point we decide whether it belongs to the
Mandelbrot set. Let $x$ be the number of points that belong to the
Mandelbrot set (by construction, $x \leq N \times N$). Let $B$ be the
area of the bounding rectangle defined as

$$
B := (\mathrm{XMAX} - \mathrm{XMIN}) \times (\mathrm{YMAX} - \mathrm{YMIN})
$$

Then, the area $A$ of the Mandelbrot set can be approximated as

$$
A \approx \frac{x}{N^2} \times B
$$

The approximation gets better as the number of points $N$ becomes
larger. The exact value of $A$ is not known, but there are [some
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

- [omp-mandelbrot-area.c](omp-mandelbrot-area.c)

***/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Higher value = slower to detect points that belong to the Mandelbrot set */
const int MAXIT = 10000;

/* Picture window size, in pixels */
const int XSIZE = 1024, YSIZE = 768;

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const double XMIN = -2.25, XMAX = 0.75;
const double YMIN = -1.5, YMAX = 1.5;

struct d_complex {
    double re;
    double im;
};

/**
 * Performs the iteration z = z*z+c, until ||z|| > 2 when point is
 * known to be outside the Mandelbrot set. If loop count reaches
 * MAXIT, point is considered to be inside the set. Returns 1 iff
 * inside the set.
 */
int inside(struct d_complex c)
{
    struct d_complex z = {0.0, 0.0}, znew;
    int it;

    for ( it = 0; (it < MAXIT) && (z.re*z.re + z.im*z.im <= 4.0); it++ ) {
        znew.re = z.re*z.re - z.im*z.im + c.re;
        znew.im = 2.0*z.re*z.im + c.im;
        z = znew;
    }
    return (it >= MAXIT);
}

int main( int argc, char *argv[] )
{
    int i, j, ninside = 0, npoints = 1000;
    double area, error;
    const double EPS = 1.0e-5;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [npoints]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        npoints = atoi(argv[1]);
    }

    printf("Using a %d x %d grid\n", npoints, npoints);

    /* Loop over grid of points in the complex plane which contains
       the Mandelbrot set, testing each point to see whether it is
       inside or outside the set. */

    const double tstart = omp_get_wtime();

#ifdef SERIAL
    /* [TODO] Parallelize the following loop(s) */
#else
    /* The "schedule(dynamic,64)" clause is here as an example only;
       the chunk size (64) might not be the best. */
#pragma omp parallel for collapse(2) default(none) shared(npoints,EPS,XMIN,XMAX,YMIN,YMAX) reduction(+:ninside) schedule(dynamic, 64)
#endif
    for (i=0; i<npoints; i++) {
        for (j=0; j<npoints; j++) {
            struct d_complex c;
            c.re = XMIN + (XMAX-XMIN)*j/npoints + EPS;
            c.im = YMIN + (YMAX-YMIN)*i/npoints + EPS;
            ninside += inside(c);
        }
    }

    const double elapsed = omp_get_wtime() - tstart;

    printf("npoints = %d, ninside = %u\n", npoints*npoints, ninside);

    /* Compute area and error estimate and output the results */
    area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/(npoints*npoints);
    error = area/npoints;

    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
    printf("Correct answer should be around 1.50659\n");
    printf("Elapsed time: %f\n", elapsed);
    return EXIT_SUCCESS;
}
