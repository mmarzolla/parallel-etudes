/****************************************************************************
 *
 * mpi-dot.c - Dot product
 *
 * Copyright (C) 2016--2023 Moreno Marzolla
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
% HPC - Dot product
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2023-11-15

The file [mpi-dot.c](mpi-dot.c) contains a MPI program that computes
the dot product between two arrays `x[]` and `y[]` of length $n$. The
dot product $s$ of two arrays `x[]` and `y[]` is defined as:

$$
s = \sum_{i = 0}^{n-1} x[i] \times y[i]
$$

In the provided program, the master performs the whole computation and
is therefore not parallel. The goal of this exercise is to write a
parallel version. Assume that, at the beginning of the program, `x[]`
and `y[]` are known only to the master. Therefore, they must be
distributed across the processes. Each process computes the scalar
product of the assigned portions of the arrays; the master then uses
`MPI_Reduce()` to sum the partial results and compute $s$.

You may initially assume that $n$ is an exact multiple of the number
of MPI processes $P$; then, relax this assumption and modify the
program so that it works with any array length $n$. The simpler
solution is to distribute the arrays using `MPI_Scatter()` and let the
master take care of any excess data. Another possibility is to use
`MPI_Scatterv()` to distribute the input unevenly across the
processes.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot -lm

To execute:

        mpirun -n P ./mpi-dot [n]

Example:

        mpirun -n 4 ./mpi-dot 1000

## Files

- [mpi-dot.c](mpi-dot.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( const double* x, const double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    const double TOL = 1e-5;
    double *x = NULL, *y = NULL, result = 0.0;
    int i, n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 == my_rank ) {
        /* The master allocates the vectors */
        x = (double*)malloc( n * sizeof(*x) ); assert(x != NULL);
        y = (double*)malloc( n * sizeof(*y) ); assert(y != NULL);
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }
#ifdef SERIAL
    /* [TODO] This is not a true parallel version, since the master
       does everything */
    if ( 0 == my_rank ) {
        result = dot(x, y, n);
    }
#else

    double *local_x, *local_y;

#ifdef USE_SCATTERV
    /* This block of code uses MPI_Scatterv() to distribute the input
       to the workers. To compile this version, use:

       mpicc -DUSE_SCATTERV -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi.dot

    */

    if (0 == my_rank) {
        printf("Computing dot product using MPI_Scatterv()\n");
    }

    int *sendcounts = (int*)malloc(comm_sz * sizeof(*sendcounts)); assert(sendcounts != NULL);
    int *displs = (int*)malloc(comm_sz * sizeof(*displs)); assert(displs != NULL);
    for (i=0; i<comm_sz; i++) {
        /* First, compute the starting and ending position of each
           block (iend is actually one element _past_ the ending
           position) */
        const int istart = n*i/comm_sz;
        const int iend = n*(i+1)/comm_sz;
        const int blklen = iend - istart;
        sendcounts[i] = blklen;
        displs[i] = istart;
    }

    const int local_n = sendcounts[my_rank]; /* how many elements I must handle */

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) ); assert(local_x != NULL);
    local_y = (double*)malloc( local_n * sizeof(*local_y) ); assert(local_y != NULL);

    /* Scatter vector x */
    MPI_Scatterv( x,             /* sendbuf             */
                  sendcounts,    /* sendcounts          */
                  displs,        /* displacements       */
                  MPI_DOUBLE,    /* sent datatype       */
                  local_x,       /* recvbuf             */
                  local_n,       /* recvcount           */
                  MPI_DOUBLE,    /* received datatype   */
                  0,             /* source              */
                  MPI_COMM_WORLD /* communicator        */
                  );

    /* Scatter vector y*/
    MPI_Scatterv( y,             /* sendbuf             */
                  sendcounts,    /* sendcounts          */
                  displs,        /* displacements       */
                  MPI_DOUBLE,    /* sent datatype       */
                  local_y,       /* recvbuf             */
                  local_n,       /* recvcount           */
                  MPI_DOUBLE,    /* received datatype   */
                  0,             /* source              */
                  MPI_COMM_WORLD /* communicator        */
                  );

    /* All nodes compute the local result */
    double local_result = dot( local_x, local_y, local_n );

    /* Reduce (sum) the local dot products */
    MPI_Reduce( &local_result,  /* send buffer          */
                &result,        /* receive buffer       */
                1,              /* count                */
                MPI_DOUBLE,     /* datatype             */
                MPI_SUM,        /* operation            */
                0,              /* destination          */
                MPI_COMM_WORLD  /* communicator         */
                );

    free(sendcounts);
    free(displs);
#else
    /* This block of code uses MPI_Scatter() to distribute the input
       to the workers; the master takes care of any leftover. To
       compile this version use:

       mpicc -std=c99 -Wall -Wpedantic mpi-dot.c -o mpi-dot

    */

    if (0 == my_rank) {
        printf("Computing dot product using MPI_Scatter()\n");
    }

    /* This version works for any value of n; the root takes care of
       the leftovers later on */
    const int local_n = n / comm_sz;

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) ); assert(local_x != NULL);
    local_y = (double*)malloc( local_n * sizeof(*local_y) ); assert(local_y != NULL);

    /* Scatter vector x */
    MPI_Scatter( x,             /* sendbuf      */
                 local_n,       /* count; how many elements to send to _each_ destination */
                 MPI_DOUBLE,    /* sent datatype */
                 local_x,       /* recvbuf      */
                 local_n,       /* recvcount    */
                 MPI_DOUBLE,    /* received datatype */
                 0,             /* source       */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* Scatter vector y*/
    MPI_Scatter( y,             /* sendbuf      */
                 local_n,       /* count; how many elements to send to _each_ destination */
                 MPI_DOUBLE,    /* sent datatype */
                 local_y,       /* recvbuf      */
                 local_n,       /* recvcount    */
                 MPI_DOUBLE,    /* received datatype */
                 0,             /* source       */
                 MPI_COMM_WORLD /* communicator */
                 );

    /* All nodes compute the local result */
    double local_result = dot( local_x, local_y, local_n );

    /* Reduce (sum) the local dot products */
    MPI_Reduce( &local_result,  /* send buffer          */
                &result,        /* receive buffer       */
                1,              /* count                */
                MPI_DOUBLE,     /* datatype             */
                MPI_SUM,        /* operation            */
                0,              /* destination          */
                MPI_COMM_WORLD  /* communicator         */
                );

    if ( 0 == my_rank ) {
        /* the master handles the leftovers, if any */
        for (i = local_n * comm_sz; i<n; i++) {
            result += x[i] * y[i];
        }
    }
#endif
    free(local_x);
    free(local_y);
#endif

    if (0 == my_rank) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < TOL ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    free(x); /* if x == NULL, does nothing */
    free(y);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
