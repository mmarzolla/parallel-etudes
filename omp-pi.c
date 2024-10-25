/****************************************************************************
 *
 * omp-pi.c - Monte Carlo approximation of PI
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Monte Carlo approximation of $\pi$
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2023-10-25

The file [omp-pi.c](omp-pi.c) implements a serial Monte Carlo
algorithm for computing the approximate value of $\pi$. Monte Carlo
algorithms use pseudo-random numbers to evaluate some function of
interest.

![Figure 1: Monte Carlo computation of the value of $\pi$](pi_Monte_Carlo.svg)

The idea is simple (see Figure 1). We generate $N$ random points
uniformly distributed over a square with corners at $(-1, -1)$ and
$(1, 1)$, and count the number $x$ of points falling inside the circle
with center $(0,0)$ and unitary radius. Then, we have:

$$
\frac{\text{N. of points inside the circle}}{\text{Total n. of points}} \approx \frac{\text{Area of circle}}{\text{Area of enclosing square}}
$$

from which, substituting the appropriate variables:

$$
\frac{x}{N} \approx \frac{\pi}{4}
$$

hence $\pi \approx 4x / N$. This estimate becomes more accurate as the
number of points $N$ increases.

The goal of this exercise is to modify the serial program to make use
of shared-memory parallelism using OpenMP.

## The hard (and inefficient) way

Start with a version that uses the `omp parallel` construct. Let $P$
be the number of OpenMP threads; then, the program operates as
follows:

1. The user specifies the number $N$ of points to generate as a
   command-line parameter, and the number $P$ of OpenMP threads using
   the `OMP_NUM_THREADS` environment variable.

2. Thread $p$ generates $N/P$ points using `generate_points()` and
   stores the result in `inside[p]`. `inside[]` is an integer array of
   length $P$ that must be declared outside the parallel region, since
   it must be shared across all OpenMP threads.

3. At the end of the parallel region, the master (thread 0) computes
   $x$ as the sum of the content of `inside[]`; from this the estimate
   of $\pi$ can be computed as above.

You may initially assume that the number of points $N$ is a multiple
of $P$; when you get a working program, relax this assumption to make
the computation correct for any value of $N$.

## The better way

A better approach is to let the compiler parallelize the "for" loop in
`generate_points()` using `omp parallel` and `omp for`.  There is a
problem, though: function `int rand(void)` is not thread-safe since it
modifies a global state variable, so it can not be called concurrently
by multiple threads. Instead, we use `int rand_r(unsigned int *seed)`
which is thread-safe but requires that each thread keeps a local
`seed`. We split the `omp parallel` and `omp for` directives, so that
a different local seed can be given to each thread like so:

```C
#pragma omp parallel default(none) shared(n, n_inside)
{
        const int my_id = omp_get_thread_num();
        \/\* Initialization of my_seed is arbitrary \*\/
        unsigned int my_seed = 17 + 19*my_id;
        ...
#pragma omp for reduction(+:n_inside)
        for (int i=0; i<n; i++) {
                \/\* call rand_r(&my_seed) here... \*\/
                ...
        }
        ...
}
```

Compile with:

        gcc -std=c99 -fopenmp -Wall -Wpedantic omp-pi.c -o omp-pi -lm

Run with:

        ./omp-pi [N]

For example, to compute the approximate value of $\pi$ using $P=4$
OpenMP threads and $N=20000$ points:

        OMP_NUM_THREADS=4 ./omp-pi 20000

## Files

- [omp-pi.c](omp-pi.c)

***/

/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
unsigned int generate_points( unsigned int n )
{
#ifdef SERIAL
    /* [TODO] parallelize the body of this function */
    unsigned int n_inside = 0;
    /* The C function `rand()` is not thread-safe, since it modifies a
       global seed; therefore, it can not be used inside a parallel
       region. We use `rand_r()` with an explicit per-thread
       seed. However, this means that in general the result computed
       by this program depends on the number of threads. */
    unsigned int my_seed = 17 + 19*omp_get_thread_num();
    for (int i=0; i<n; i++) {
        /* Generate two random values in the range [-1, 1] */
        const double x = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        const double y = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
        if ( x*x + y*y <= 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
#else
#if 0
    /* This version uses neither the "parallel for" nor the
       "reduction" directives. It is instructive to try to parallelize
       the "for" loop by hand, but in practice you should never do
       that unless there are specific reasons. */
    const int n_threads = omp_get_max_threads();
    unsigned int my_n_inside[n_threads];

#pragma omp parallel num_threads(n_threads) default(none) shared(n, my_n_inside, n_threads)
    {
        const int my_id = omp_get_thread_num();
        /* We make sure that exactly `n` points are generated. Note
           that the right-hand side of the assignment can NOT be
           simplified algebraically, since the '/' operator here is
           the truncated integer division and a/c + b/c != (a+b)/c
           (e.g., a=5, b=5, c=2, a/c + b/c == 4, (a+b)/c == 5). */
        const unsigned int local_n = (n*(my_id + 1))/n_threads - (n*my_id)/n_threads;
        unsigned int my_seed = 17 + 19*my_id;
        my_n_inside[my_id] = 0;
        for (int i=0; i<local_n; i++) {
            /* Generate two random values in the range [-1, 1] */
            const double x = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
            const double y = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
            if ( x*x + y*y <= 1.0 ) {
                my_n_inside[my_id]++;
            }
        }
    } /* end of the parallel region */
    unsigned int n_inside = 0;
    for (int i=0; i<n_threads; i++) {
        n_inside += my_n_inside[i];
    }
    return n_inside;
#else
    unsigned int n_inside = 0;
    /* This is a case where it is necessary to split the "omp
       parallel" and "omp for" directives. Indeed, each thread uses a
       private `my_seed` variable to keep track of the seed of the
       pseudo-random number generator. The simplest way to create such
       variable is to first create a parallel region, and define a
       local (private) variable `my_seed` before using the `omp for`
       construct to execute the loop in parallel. */
#pragma omp parallel default(none) shared(n, n_inside)
    {
        const int my_id = omp_get_thread_num();
        unsigned int my_seed = 17 + 19*my_id;
#pragma omp for reduction(+:n_inside)
        for (int i=0; i<n; i++) {
            /* Generate two random values in the range [-1, 1] */
            const double x = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
            const double y = (2.0 * rand_r(&my_seed)/(double)RAND_MAX) - 1.0;
            if ( x*x + y*y <= 1.0 ) {
                n_inside++;
            }
        }
    } /* end of the parallel region */
    return n_inside;
#endif
#endif
}

int main( int argc, char *argv[] )
{
    unsigned int n_points = 10000;
    unsigned int n_inside;
    const double PI_EXACT = 3.14159265358979323846;

    if ( argc > 2 ) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        n_points = atol(argv[1]);
    }

    printf("Generating %u points...\n", n_points);
    const double tstart = omp_get_wtime();
    n_inside = generate_points(n_points);
    const double elapsed = omp_get_wtime() - tstart;
    const double pi_approx = 4.0 * n_inside / (double)n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0*fabs(pi_approx - PI_EXACT)/PI_EXACT);
    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
