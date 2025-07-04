/****************************************************************************
 *
 * omp-mandelbrot-area.c - Area of the Mandelbrot set
 *
 * Copyright (C) 2017--2024 Moreno Marzolla
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
% Area of the Mandelbrot set
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-10-11

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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

/* Higher value = slower to detect points that belong to the Mandelbrot set */
const int MAXIT = 10000;

/* We consider the region on the complex plane -2.25 <= Re <= 0.75
   -1.4 <= Im <= 1.5 */
const float XMIN = -2.25, XMAX = 0.75;
const float YMIN = -1.4, YMAX = 1.5;

/**
 * Performs the iteration z = z*z+c, until ||z|| > 2 when point is
 * known to be outside the Mandelbrot set. Return the number of
 * iterations until ||z|| > 2, or MAXIT.
 */
int iterate(float cx, float cy)
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
    /* [TODO] Parallelize the following loop(s) */
#else
    /* The "schedule(dynamic,64)" clause is here as an example only;
       the chunk size (64) might not be the best. */
#pragma omp parallel for collapse(2) default(none) shared(xsize,ysize,XMIN,XMAX,YMIN,YMAX,MAXIT) reduction(+:ninside) schedule(dynamic, 64)
#endif
    for (int i=0; i<ysize; i++) {
        for (int j=0; j<xsize; j++) {
            const float cx = XMIN + (XMAX-XMIN)*j/xsize;
            const float cy = YMIN + (YMAX-YMIN)*i/ysize;
            const int it = iterate(cx, cy);
            ninside += (it >= MAXIT);
        }
    }
    return ninside;
}

int main( int argc, char *argv[] )
{
    int N = 1000;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("Using a %d x %d grid\n", N, N);

    /* Loop over grid of points in the complex plane which contains
       the Mandelbrot set, testing each point to see whether it is
       inside or outside the set. */

    const double tstart = omp_get_wtime();
    const uint32_t ninside = inside(N, N);
    const double elapsed = omp_get_wtime() - tstart;

    printf("N = %d, ninside = %u\n", N*N, ninside);

    /* Compute area and output the results */
    const float area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/(N*N);

    printf("Area of Mandlebrot set = %f\n", area);
    printf("Correct answer should be around 1.50659\n");
    printf("Elapsed time: %f\n", elapsed);
    return EXIT_SUCCESS;
}
