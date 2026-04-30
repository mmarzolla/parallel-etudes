/****************************************************************************
 *
 * omp-circles.c - Monte Carlo estimation of the area of the union of circles
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
% Monte Carlo estimation of the area of the union of circles
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-30

The file [omp-circles.c](omp-circles.c) contains a serial
implementation of a Monte Carlo algorithm that estimates the area of
the union of $N$ circles. Let `cx[i]`, `cy[i]`, and `r[i]` be the
coordinates of the center of circle $i$ and its radius.  All circles
are entirely contained within the bounding square with opposites
corners $(0, 0)$ and $(1000, 1000)$.

Circles may overlap in whole or in part; therefore, it is not easy to
compute the area of their union. We implement a Monte Carlo algorithm
to estimate the area; the idea is similar to the estimation of the
value of $\pi$ by generating random points, and is as follows:

- Generate $K$ random points uniformly distributed inside the bounding
  square $(0, 0)$, $(1000, 1000)$. Let $c$ be the number of points
  that fall within at least one circle.

- The area $A$ of the union of the circles can be estimated as $A
  \approx 1000 \times 1000 \times c/K$. In other words, the area $A$
  is the product of the area of the bounding square and the fraction
  of points $c/K$ that falls within at least one circle.

![Figure 1: Monte Carlo estimation of the area of ​​the union of
 circles](mpi-circles.svg)

Figure 1 illustrates the idea.

The purpose of this exercise is to distribute the computation among a
team of OpenMP threads. Each thread should generate $K/P$ points and
test each point with all the circles; the total number of points
inside the circles is the sum-reduction of the partial counts.

A critical issue with this program is the generation of random
numbers.  The function `rand()` is not thread-safe, since it might
store current state of the generator in a global
variable[^1]. Function `int rand_r(unsigned int *seed)` is a
thread-safe version of `rand()`. Since the sequence of pseudo-random
numbers depends on the seed, each OpenMP thread must initialize the
seed to a different value.

Random number generation is a though issue. The use of `rand()` and
`rand_r()` is not recommended for serious applications. Furthermore,
parallel random number generation raises a new set of issues that we
can not address here. For example, the parallel version of this
program generates a sequence of random points which is not the same as
the serial program; furthermore, the sequence will be different
depending on the number of OpenMP threads that are created.

[^1]: The man page of `rand()` shipped with Ubuntu 24.04 contains
      contradicting information. It states that function `rand()` is
      not reentrant (i.e., is not thread-safe), but later on states
      that `rand()` is thread safe. The `rand()` implementation in the
      latest [glibc](https://www.gnu.org/software/libc/) appears
      thread-safe indeed, since there is a mutex inside the function.

To compile:

        gcc -fopenmp -std=c99 -Wall -Wpedantic omp-circles.c -o omp-circles

To execute:

        ./omp-circles N input_file_name

Example:

        ./omp-circles 10000 circles-1000.in

## Files

- [omp-circles.c](omp-circles.c)
- [gen-circles.c](gen-circles.c) (to generate random inputs)
- [circles-1000.in](circles-1000.in)
- [circles-10000.in](circles-10000.in)

***/
#if _POSIX_C_SOURCE < 199506L
/* required for rand_r() */
#define _POSIX_C_SOURCE 199506L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Computes the square of x. */
float sq(float x)
{
    return x*x;
}

/* Generate a random float in [a, b]. This function is thread-safe
   since it uses rand_r(); the seed must be provided as parameter. */
float randab(float a, float b, unsigned int *seed)
{
    return a + (b-a)*(rand_r(seed) / (float)RAND_MAX);
}

/* Generate `k` random points inside the square (0,0) --
  (100,100). Return the number of points that fall inside at least one
  of the `n` circles with center (x[i], y[i]) and radius r[i]. The
  result must be <= k. */
int inside( const float* x, const float* y, const float *r, int n, int k )
{
    int ninside=0;
#ifdef SERIAL
    unsigned int seed = 42;
#else
#pragma omp parallel default(none) shared(k, x, y, r, n) reduction(+:ninside)
    {
        unsigned int seed = 42 + omp_get_thread_num() * 13;
#pragma omp for
#endif
    for (int np=0; np<k; np++) {
        const float px = randab(0.0, 100.0, &seed);
        const float py = randab(0.0, 100.0, &seed);
        /* Iterate over the circles. */
        for (int i=0; i<n; i++) {
            if ( sq(px-x[i]) + sq(py-y[i]) <= sq(r[i]) ) {
                ninside++;
                break;
            }
        }
    }
#ifndef SERIAL
    }
#endif
    return ninside;
}

int main( int argc, char* argv[] )
{
    int N, K;

    if ( argc != 3 ) {
        fprintf(stderr, "Usage: %s [npoints] [inputfile]\n", argv[0]);
        return EXIT_FAILURE;
    }

    K = atoi(argv[1]);

    FILE *in = fopen(argv[2], "r");
    int i;
    if ( in == NULL ) {
        fprintf(stderr, "FATAL: Cannot open \"%s\" for reading\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (1 != fscanf(in, "%d", &N)) {
        fprintf(stderr, "FATAL: Cannot read the number of circles\n");
        return EXIT_FAILURE;
    }
    float *x = (float*)malloc(N * sizeof(*x)); assert(x != NULL);
    float *y = (float*)malloc(N * sizeof(*y)); assert(y != NULL);
    float *r = (float*)malloc(N * sizeof(*r)); assert(r != NULL);
    for (i=0; i<N; i++) {
        if (3 != fscanf(in, "%f %f %f", &x[i], &y[i], &r[i])) {
            fprintf(stderr, "FATAL: Cannot read circle %d\n", i);
            return EXIT_FAILURE;
        }
    }
    fclose(in);

    const double tstart = omp_get_wtime();
    const int c = inside(x, y, r, N, K);
    const double elapsed = omp_get_wtime() - tstart;

    printf("%d points, %d inside, area=%f\n", K, c, 1.0e6*c/K);
    printf("Execution time %.3f\n", elapsed);

    free(x);
    free(y);
    free(r);

    return EXIT_SUCCESS;
}
