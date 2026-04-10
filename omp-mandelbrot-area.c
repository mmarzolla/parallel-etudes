/****************************************************************************
 *
 * omp-mandelbrot-area.c - Area of the Mandelbrot set
 *
 * Copyright (C) 2017--2026 Moreno Marzolla
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
% Last updated: 2026-04-10

The Mandelbrot set is the set of black points in Figure 1.

![Figure 1: The Mandelbrot set.](mandelbrot-set.png)

The file [omp-mandelbrot-area.c](omp-mandelbrot-area.c) contains a
serial program that estimates the area of the Mandelbrot set using a
Monte Carlo method.

The program works as follows. First, it defines a rectangle in the
complex plane that contains the Mandelbrot set. Let _(XMIN, YMIN)_ and
_(XMAX, YMAX)_ be the upper left and lower right coordinates of such a
rectangle. Then, the program generates $N$ random points inside the
rectangle. Let $x$ be the number of points that belong to the
Mandelbrot set (by construction, $x \leq N$). Let $B$ be the area of
the bounding rectangle defined as

$$
B := (\mathrm{XMAX} - \mathrm{XMIN}) \times (\mathrm{YMAX} - \mathrm{YMIN})
$$

The area $A$ of the Mandelbrot set can be approximated as

$$
A \approx \frac{x}{N} \times B
$$

The approximation gets better as the number of points $N$ becomes
larger. The exact value of $A$ is not known, but there are [some
estimates](https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html).

Modify the serial program to use the shared-memory parallelism
provided by OpenMP. If there are $P$ OpenMP processes, each one
generates $N/P$ points (care should be taken if $N$ is not an integer
multiple of $P$). The result is the sum-reduction of the partial
counts from each thread. This can be achieved with the `reduction`
clause.

The tricky part is the generation of random numbers. According to the
documentation, `rand()` is not thread-safe because it may use a global
variable to store the state of the random number generator. To address
this issue it is necessary to use the function `int rand_r(unsigned
int *seedp)` where the state is explicitly passed as
parameter[^1]. Therefore, each OpenMP thread needs to instantiate a
local state whose initial value should depend on the thread ID, to
guarantee repeatable results.

[^1]: The man page for `rand_r()` states that it is deprecated.

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

/**
 * Return a random value in [a, b]. This function uses `rand_r()`, so
 * the state of the pseudo-random number generator must be passed
 * explicitly.
 */
float randab(float a, float b, unsigned int *state)
{
    return a + (b-a) * (rand_r(state) / (float)RAND_MAX);
}

int main( int argc, char *argv[] )
{
    long N = 1000000l;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N = atol(argv[1]);
    }

    printf("Generating %ld points\n", N);

    const double tstart = omp_get_wtime();
    long ninside = 0;
#ifdef SERIAL
    /* [TODO] Parallelize the following loop */
    unsigned int state = 1;
#else
#pragma omp parallel default(none) shared(XMIN, XMAX, YMIN, YMAX, MAXIT, N) reduction(+:ninside)
    {
    const int my_id = omp_get_thread_num();
    unsigned int state = 17 + my_id * 3;
#pragma omp for schedule(dynamic,32)
#endif
    for (long i=0; i<N; i++) {
        const float cx = randab(XMIN, XMAX, &state);
        const float cy = randab(YMIN, YMAX, &state);
        const int it = iterate(cx, cy);
        ninside += (it >= MAXIT);
    }
#ifndef SERIAL
    }
#endif
    const double elapsed = omp_get_wtime() - tstart;

    printf("N = %ld, ninside = %ld\n", N, ninside);

    /* Compute area and output the results */
    const float area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/N;

    printf("Area of Mandlebrot set = %f\n", area);
    printf("Correct answer should be around 1.50659\n");
    printf("Execution time %.3f\n", elapsed);
    return EXIT_SUCCESS;
}
