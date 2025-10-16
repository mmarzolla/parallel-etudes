/****************************************************************************
 *
 * mpi-sum.c - Sum-reduction of an array
 *
 * Copyright (C) 2018--2022 Moreno Marzolla
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
% Sum-reduction of an array
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2022-10-27

The file [mpi-sum.c](mpi-sum.c) contains a serial implementation of an
MPI program that computes the sum of an array of length $N$; indeed,
the program performsa a _sum-reduction_ of the array. In the version
provided, process 0 performs all computations, and therefore is not a
true parallel program. Modify the program so that all processes
contribute to the computation according to the following steps (see
Figure 1).

![Figure 1: Computing the sum-reduction using MPI.](mpi-sum.svg)

1. The master process (rank 0) creates and initializes the input array
   `master_array[]`.

2. The master distributes `master_array[]` array among the $P$
   processes (including itself) using `MPI_Scatter()`. You may
   initially assume that $N$ is an integer multiple of $P$.

3. Each process computes the sum-reduction of its portion.

4. Each process $p > 0$ sends its own local sum to the master using
   `MPI_Send()`; the master receives the local sums using `MPI_Recv()`
   and accumulates them.

We will see in the next lexture how step 4 can be realized more
efficiently with the MPI reduction operation.

If the array length $N$ is not a multiple of $P$, there are several
possible solutions:

a. _Padding_: add extra elements to the array so that the new length
   $N'$ is multiple of $P$. The extra elements must be initialized to
   zero, so that the sum does not change. This solution requires that
   the procedure has some control on the generation of the input
   array, so that the length can be changed. This is not always
   possible nor desirable, e.g., if the sum-reduction must be
   implemented as a subroutine that receives the input as a parameter
   over which the subroutine has no control.

b. _Use Scatterv_: MPI provides the `MPI_Scatterv` function which
   works like `MPI_Scatter` but allows different block sizes. The
   downside is that `MPI_Scatterv` is cumbersome to use because it
   requires array parameters that must be allocated/deallocated and
   properly initialized.

c. _Let the master handle the leftover_: if $N$ is not an integer
   multiple of $P$, the master (rank 0) takes care of the leading or
   trailing `N % P` elements, in addition to its own block of length
   $N/P$ like any other process. The limitation of this approach is
   that it introduces some load imbalance: by definition, `N % P` is a
   number between $0$ and $P-1$ inclusive, which may be significant if
   the number of processes $P$ is large and/or the computation time is
   heavily influenced by the chunk sizes.

For this exercise I suggest option c: in our setting, $P$ is small due
to hardware limitations and the computation is trivial. Hence, the
execution time is likely to be dominated by the communication
operations anyway.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-sum.c -o mpi-sum

To execute:

        mpirun -n P ./mpi-sun N

Example:

        mpirun -n 4 ./mpi-sum 10000

## Files

- [mpi-sum.c](mpi-sum.c)

***/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Compute the sum of all elements of array `v` of length `n` */
float sum(float *v, int n)
{
    float sum = 0;
    int i;

    for (i=0; i<n; i++) {
        sum += v[i];
    }
    return sum;
}

/* Fill the array array `v` of length `n`; return the sum of the
   content of `v` */
float fill(float *v, int n)
{
    const float vals[] = {1, -1, 2, -2, 0};
    const int NVALS = sizeof(vals)/sizeof(vals[0]);

    for (int i=0; i<n; i++) {
        v[i] = vals[i % NVALS];
    }
    switch(n % NVALS) {
    case 1: return 1; break;
    case 3: return 2; break;
    default: return 0;
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    float *master_array = NULL, s = 0, expected = 0.0f;;
    int n = 10000;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    /* The master initializes the array */
    if ( 0 == my_rank ) {
        master_array = (float*)malloc( n * sizeof(float) );
        assert(master_array != NULL);
        expected = fill(master_array, n);
    }

#ifdef SERIAL
    if ( 0 == my_rank ) {
        /* [TODO] This is not a true parallel version; the master does
           everything */
        s = sum(master_array, n);
    }
#else
    const int local_n = n/comm_sz; /* is n is not an integer multiple
                                      of comm_sz, there will be some
                                      leftovers. The master will take
                                      care of them later on */
    float* local_array = (float*)malloc( local_n * sizeof(float) );
    assert(local_array != NULL);

    MPI_Scatter( master_array,  /* sendbuf      */
                 local_n,       /* sendcount    */
                 MPI_FLOAT,     /* sendtype     */
                 local_array,   /* recvbuf      */
                 local_n,       /* recvcount    */
                 MPI_FLOAT,     /* rcvtype      */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );


    /* Compute the local sum */
    const float local_sum = sum(local_array, local_n);

    free(local_array);

    if ( 0 == my_rank ) {
        /* Take care of any leftover

                                                 +-- leftovers
                                                 v
           +--------+--------+--------+--------+---+
           |        |        |        |        |///|
           +--------+--------+--------+--------+---+
             local_n
            elements
           \-----------------------------------/
                 local_n * comm_sz elements

         */
        s = sum(master_array + local_n * comm_sz, n % comm_sz);

        /* Get the sums from other processes (should be better done with MPI_Reduce */
        float remote_sum;
        s += local_sum;
        for (int p=1; p<comm_sz; p++) {
            MPI_Recv( &remote_sum,      /* buf          */
                      1,                /* size         */
                      MPI_FLOAT,        /* datatype     */
                      MPI_ANY_SOURCE,   /* source       */
                      MPI_ANY_TAG,      /* tag          */
                      MPI_COMM_WORLD,   /* comm         */
                      MPI_STATUS_IGNORE /* status       */
                      );
            /* printf("Received %f from proc %d\n", remote_sum, status.MPI_SOURCE); */
            s += remote_sum;
        }
    } else {
        /* Send local sum to the master */
        MPI_Send( &local_sum,           /* buf          */
                  1,                    /* size         */
                  MPI_FLOAT,            /* datatype     */
                  0,                    /* dest         */
                  0,                    /* tag          */
                  MPI_COMM_WORLD        /* comm         */
                  );
    }
#endif
    free(master_array);

    if (0 == my_rank) {
        printf("Sum=%f, expected=%f\n", s, expected);
        if (s == expected) {
            printf("Test OK\n");
        } else {
            printf("Test FAILED\n");
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
