/****************************************************************************
 *
 * mpi-mandelbrot-area.c - Area of the Mandelbrot set
 *
 * Copyright (C) 2024 by Moreno Marzolla <moreno.marzolla@unibo.it>
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
% HPC - Area of the Mandelbrot set
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2024-01-05

![](mandelbrot-set.png)

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot-area.c -o mpi-mandelbrot-area -lm

To execute:

        mpirun -n NPROC ./mpi-mandelbrot-area [npoints]

Example:

        mpirun -n 4 ./mpi-mandelbrot-area 1000

## Files

- [mpi-mandelbrot-area.c](mpi-mandelbrot-area.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

/* Higher value = slower to detect points that belong to the Mandelbrot set */
const int MAXIT = 10000;

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

/* Generate a random number in [a, b] */
double randab(double a, double b)
{
    return a + ((b-a)*rand())/RAND_MAX;
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int npoints;
    int ninside = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        npoints = atoi(argv[1]);
    } else {
        npoints = 1000;
    }

    srand(123 + 7*my_rank);

    const double tstart = MPI_Wtime();
#ifdef SERIAL
    if ( 0 == my_rank ) {
        /* [TODO] This is not a true parallel version, since the master
           does everything */
        for (int i=0; i<npoints; i++) {
            struct d_complex c;
            c.re = randab(XMIN, XMAX);
            c.im = randab(YMIN, YMAX);
            ninside += inside(c);
        }
    }
#else
    const int local_npoints = npoints / comm_sz + (my_rank < npoints % comm_sz);
    int local_ninside = 0;
    for (int i=0; i<local_npoints; i++) {
        struct d_complex c;
        c.re = randab(XMIN, XMAX);
        c.im = randab(YMIN, YMAX);
        local_ninside += inside(c);
    }
    printf("Rank=%d local_npoints=%d local_ninside=%d\n", my_rank, local_npoints, local_ninside);
    MPI_Reduce(&local_ninside,  /* sendbuf      */
               &ninside,        /* recfbuf      */
               1,               /* count        */
               MPI_INT,         /* datatype     */
               MPI_SUM,         /* op           */
               0,               /* root         */
               MPI_COMM_WORLD);
#endif
    const double elapsed = MPI_Wtime() - tstart;

    if (0 == my_rank) {
        printf("npoints = %d, ninside = %u\n", npoints, ninside);

        /* Compute area and error estimate and output the results */
        const double area = (XMAX-XMIN)*(YMAX-YMIN)*ninside/npoints;
        const double error = area/sqrt(npoints);

        printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
        printf("Correct answer should be around 1.50659\n");
        printf("Elapsed time: %f\n", elapsed);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
