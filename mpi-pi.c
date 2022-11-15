/****************************************************************************
 *
 * mpi-pi.c - Monte Carlo approximatino of PI
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Monte Carlo approximation of PI
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-24

The file [mpi-pi.c](mpi-pi.c) contains a serial program for computing
the approximate value of $\pi$ using a Monte Carlo algorithm. Monte
Carlo algorithms use pseudorandom numbers to compute an approximation
of some quantity of interest.

![Figure 1: Monte Carlo computation of the value of $\pi$](pi_Monte_Carlo.png)

The idea is quite simple (see Figure 1). We generate $N$ random points
uniformly distributed inside the square with corners at $(-1, -1)$ and
$(1, 1)$. Let $x$ be the number of points that lie inside the circle
inscribed in the square; then, the ratio $x / N$ is an approximation
of the ratio between the area of the circle and the area of the
square. Since the area of the circle is $\pi$ and the area of the
square is $4$, we have $x/N \approx \pi / 4$ which yelds $\pi \approx
4x / N$. This estimate becomes more accurate as the number of points
$N$ increases.

Modify the serial program to parallelize the computation. Several
parallelization strategies are possible, but for now you are advised
to implement the following one ($P$ is the number of MPI processes):

1. Each process gets the value of the number of points $N$ from the
   command line; you may initially assume that $N$ is a multiple of
   $P$, and then relax this requirement to make the program with any
   value of $N$.

2. Each process $p$, including the master, generates $N/P$ random
   points and keeps track of the number $x_p$ of points inside the
   circle;

3. Each process $p > 0$ sends its local value $x_p$ to the master
   using point-to-point send/receive operations.

4. The master receives $x_p$ from all each process $p = 1, \ldots,
   P-1$ (the master already knows $x_0$), computes their sum $x$ and
   the prints the approximate value of $\pi$ as $(4x / N)$.

Step 3 above should be performed using send/receive
operations. However, note that **this is not efficient** and should be
done using the MPI reduction operation that will be introduced in the
next lecture.

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
    int i, n_inside = 0;
    for (i=0; i<n; i++) {
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

    /* This is what we can do using send/receive only. The correct
       solution would be to use MPI_Reduce() */
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
