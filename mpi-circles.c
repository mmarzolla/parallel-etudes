/****************************************************************************
 *
 * mpi-circles.c - Monte Carlo estimation of the area of the union of circles
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Monte Carlo estimation of the area of the union of circles
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-13

The ile [mpi-circles.c](mpi-circles.c) contains a serial
implementation of a Monte Carlo algorithm that estimates the area of
​​the union of $N$ circles. Let `cx[i]`, `cy[i]`, and `r[i]` be the
coordinates of the center of circle $i$ and its radius. Assume that
all circles are entirely contained within the bounding square with
opposites corners $(0, 0)$ and $(1000, 1000)$.

Circles may overlap in whole or in part; therefore, it is not easy to
determine the area of their union. We implement a _Monte Carlo_
algorithm to estimate the area; the idea is similar to the estimation
of the value of $\pi$ by generating random points, and is as follows:

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

The file [mpi-circles.c](mpi-circles.c) contains a serial program
where process 0 performs the whole computation. The purpose of this
exercise is to distribute the computation among all MPI processes. The
input file must be read by process 0 only; therefore, only process 0
knows the number $N$ of circles and their coordinates, so it must send
all required information to the other processes. The program must work
correctly for any value of $N$, $K$ and number of MPI processes $P$.

To do so, each process $p$ generates $K/P$ points and test each point
with _all_ the circles and computes the number of points $C_p$ that
fall inside at least one circle. The master computes the sum $C =
\sum_{p=0}^{P-1} C_p$ using a reduction, and estimates the area using
the formula given above. Therefore, each process must receive a full
copy of the arrays `cx[]`, `cy[]` and `r[]`.

> **Note:** You might be tempted to distribute the circles among the
> MPI processes, so that each process handles $N/P$ circles. This
> would not work: why?

To compile:

        mpicc -std = c99 -Wall -Wpedantic mpi-circles.c -o mpi-circles

To execute:

        mpirun -n P ./mpi-circles N input_file_name

Example:

        mpirun -n 4 ./mpi-circles 10000 circles-1000.in

## Files

- [mpi-circles.c](mpi-circles.c)
- [circles-gen.c](circles-gen.c) (to generate random inputs)
- [circles-1000.in](circles-1000.in)
- [circles-10000.in](circles-10000.in)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/* Computes the square of x */
float sq(float x)
{
    return x*x;
}

/* Generate `k` random points inside the square (0,0) --
  (100,100). Return the number of points that fall inside at least one
  of the `n` circles with center (x[i], y[i]) and radius r[i].  The
  result must be <= k. */
int inside( const float* x, const float* y, const float *r, int n, int k )
{
    int i, np, c=0;
    for (np=0; np<k; np++) {
        const float px = 100.0*rand()/(float)RAND_MAX;
        const float py = 100.0*rand()/(float)RAND_MAX;
        for (i=0; i<n; i++) {
            if ( sq(px-x[i]) + sq(py-y[i]) <= sq(r[i]) ) {
                c++;
                break;
            }
        }
    }
    return c;
}

int main( int argc, char* argv[] )
{
    float *x = NULL, *y = NULL, *r = NULL;
    int N, K, c = 0;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Initialize the Random Number Generator (RNG) */
    srand(my_rank * 7 + 11);

    if ( (0 == my_rank) && (argc != 3) ) {
        fprintf(stderr, "Usage: %s [npoints] [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    K = atoi(argv[1]);

    /* The input file is read by the master only */
    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[2], "r");
        int i;
        if ( in == NULL ) {
            fprintf(stderr, "FATAL: Cannot open \"%s\" for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (1 != fscanf(in, "%d", &N)) {
            fprintf(stderr, "FATAL: Cannot read the number of circles\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        x = (float*)malloc(N * sizeof(*x)); assert(x != NULL);
        y = (float*)malloc(N * sizeof(*y)); assert(y != NULL);
        r = (float*)malloc(N * sizeof(*r)); assert(r != NULL);
        for (i=0; i<N; i++) {
            if (3 != fscanf(in, "%f %f %f", &x[i], &y[i], &r[i])) {
                fprintf(stderr, "FATAL: Cannot read circle %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        fclose(in);
    }

    const double tstart = MPI_Wtime();

#ifdef SERIAL
    /* [TODO] This is not a true parallel version: the master does
       everything */
    if ( 0 == my_rank ) {
        c = inside(x, y, r, N, K);
    }
#else
    /* Broadcast the number of circles N, so that all processes can
       allocate the proper space for vectors x[], y[] and r[] */
    MPI_Bcast( &N,              /* buffer       */
               1,               /* count        */
               MPI_INT,         /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* communicator */
               );

    /* The master process already has x[], y[] and r[] allocated */
    if ( my_rank > 0 ) {
        x = (float*)malloc(N * sizeof(*x)); assert(x != NULL);
        y = (float*)malloc(N * sizeof(*y)); assert(y != NULL);
        r = (float*)malloc(N * sizeof(*r)); assert(r != NULL);
    }

    /* The master broadcasts a copy of the data of all circles to all
       processes */
    MPI_Bcast( x,               /* buffer       */
               N,               /* count        */
               MPI_FLOAT,       /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* communicator */
               );

    MPI_Bcast( y,               /* buffer       */
               N,               /* count        */
               MPI_FLOAT,       /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* communicator */
               );

    MPI_Bcast( r,               /* buffer       */
               N,               /* count        */
               MPI_FLOAT,       /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* communicator */
               );

    int local_K = K / comm_sz;

    /* the master handles any excess points */
    if ( 0 == my_rank )
        local_K += K % comm_sz;

    int local_c = inside(x, y, r, N, local_K);

    MPI_Reduce( &local_c,       /* sendbuf      */
                &c,             /* recvbuf      */
                1,              /* count        */
                MPI_INT,        /* datatype     */
                MPI_SUM,        /* operation    */
                0,              /* root         */
                MPI_COMM_WORLD  /* comm         */
                );
#endif

    /* the master prints the result */
    if ( 0 == my_rank ) {
        printf("%d points, %d inside, area=%f\n", K, c, 1.0e6*c/K);
        const double elapsed = MPI_Wtime() - tstart;
        printf("Execution time (s): %f\n", elapsed);
    }

    free(x);
    free(y);
    free(r);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
