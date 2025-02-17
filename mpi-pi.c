/****************************************************************************
 *
 * mpi-pi.c - Monte Carlo approximatino of PI
 *
 * Copyright (C) 2017--2022, 2024 Moreno Marzolla
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
% HPC - Monte Carlo approximation of PI
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-10-25

The file [mpi-pi.c](mpi-pi.c) contains a serial program for computing
the approximate value of $\pi$ using a Monte Carlo algorithm. Monte
Carlo algorithms use pseudorandom numbers to compute an approximation
of some quantity of interest.

![Figure 1: Monte Carlo computation of the value of $\pi$](pi_Monte_Carlo.svg)

The idea is simple (see Figure 1). We generate $N$ random points
uniformly distributed over a square with corners at $(-1, -1)$ and
$(1, 1)$, and count the number $I$ of points falling inside the circle
with center $(0,0)$ and unitary radius. Then, we have:

$$
\frac{\text{N. of points inside the circle}}{\text{Total n. of points}} \approx \frac{\text{Area of circle}}{\text{Area of enclosing square}}
$$

from which, substituting the appropriate variables:

$$
\frac{I}{N} \approx \frac{\pi}{4}
$$

hence $\pi \approx 4 I / N$. This estimate becomes more accurate as the
number of points $N$ increases.

Modify the serial program to parallelize the computation. Several
parallelization strategies are possible, but for now you are advised
to implement the following one ($P$ is the number of MPI processes):

1. Each process gets the value of the number of points $N$ from the
   command line. You may initially assume that $N$ is a multiple of
   $P$, and then relax this requirement to make the program with any
   value of $N$.

2. Each process $p$, including the master, generates $N/P$ random
   points and keeps track of the number $I_p$ of points that fall
   inside the circle;

3. The master computes the total number $I$ of points that fall inside
   the circle as the sum of $I_p$, $p=0, \ldots, P-1$.

Step 3 involves a reduction operation. Start by implementing the
inefficient solution, e.g., each process $p > 0$ sends its local value
$I_p$ to the master using point-to-point send/receive operations. The
master receives $I_p$ from all each process $p = 1, \ldots, P-1$ (the
master already knows $I_0$), computes their sum $I$ and the prints the
approximate value of $\pi$ as $(4 I / N)$.

Once you have a working implementation, modify it to use the preferred
solution, i.e., `MPI_Reduce()` instead of point-to-point
communications. 

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-pi.c -o mpi-pi -lm

To execute:

        mpirun -n P ./mpi-pi [N]

Example, using 4 MPI processes:

        mpirun -n 4 ./mpi-pi 1000000

## Files

- [mpi-pi.c](mpi-pi.c)

***/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <math.h>   /* for fabs() */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
int generate_points( int n )
{
    int n_inside = 0;
    for (int i=0; i<n; i++) {
        const double x = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        const double y = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        if ( x*x + y*y < 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int inside = 0, npoints = 1000000;
    double pi_approx;
#ifndef SERIAL
    int local_n, local_inside;
#endif

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        npoints = atoi(argv[1]);
    }

    /* Each process initializes the pseudo-random number generator; if
       we don't do this (or something similar), each process would
       produce the exact same sequence of pseudo-random numbers! */
    srand(my_rank * 11 + 7);

#ifdef SERIAL
    /* [TODO] This is not a true parallel version; the master does
       everything */
    if ( 0 == my_rank ) {
        inside = generate_points(npoints);
        pi_approx = 4.0 * inside / (double)npoints;
        printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n", pi_approx, M_PI, 100.0*fabs(pi_approx-M_PI)/M_PI);
    }
#else
    local_n = npoints / comm_sz;

    /* The master handles the leftovers */
    if ( 0 == my_rank ) {
        local_n += npoints % comm_sz;
    }

    /* All processes compute how many points are inside the circle */
    printf("Proc %d generates %d points...\n", my_rank, local_n);
    local_inside = generate_points(local_n);

    /* The solution below is NOT efficient since it relies on
       send/receive operations to accumulate the values at the
       master. The correct solution is to use MPI_Reduce() */
    if (my_rank > 0) {
        /* All processes, except the master, send the local count to
           proc 0 */
        MPI_Send( &local_inside,        /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  0,                    /* dest         */
                  0,                    /* tag          */
                  MPI_COMM_WORLD        /* communicator */
                  );
    } else {
        /* The master performs the reduction */
        inside = local_inside;
        int tmp;
        for (int i=1; i<comm_sz; i++) {
            MPI_Recv( &tmp,             /* buf          */
                      1,                /* count        */
                      MPI_INT,          /* datatype     */
                      MPI_ANY_SOURCE,   /* source       */
                      MPI_ANY_TAG,      /* tag          */
                      MPI_COMM_WORLD,   /* communicator */
                      MPI_STATUS_IGNORE /* status       */
                      );
            inside += tmp;
        }
    }
    if ( 0 == my_rank ) {
        pi_approx = 4.0 * inside / (double)npoints;
        printf("PI approximation is %f (true value=%f, rel error=%.3f%%)\n", pi_approx, M_PI, 100.0*fabs(pi_approx-M_PI)/M_PI);
    }
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
