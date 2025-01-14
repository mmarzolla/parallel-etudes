/****************************************************************************
 *
 * mpi-check-sequence.c - Check a sequence
 *
 * Copyright (C) 2025 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Check a sequence
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2025-01-13

Write an MPI program that checks whether an integer array `v[]` of
length `n` contains the values `{0, 1, ... n-1}` in that order.

Assume that:

- The array `v[]` is initially known by process 0 only.

- The length of `v[]` is a multiple of the number _P_ of processes.

- At the end, process 0 should know the result.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-check-sequence.c -o mpi-check-sequence

To execute:

        mpirun -n P ./mpi-check-sequence [N [v]]

where `N` is the array length, `v` is true (nonzero) if the sequence
must be valid, 0 otherwise.

Example:

        mpirun -n 4 ./mpi-check-sequence

## Files

- [mpi-check-sequence.c](mpi-check-sequence.c)

***/
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <assert.h>
#include <mpi.h>

void fill(int *v, int n, int must_be_valid)
{
    for (int i=0; i<n; i++) {
        v[i] = i;
    }
    if (!must_be_valid) {
        int idx = rand() % n;
        v[idx] = -1;
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int n = 1000;       /* array length */
    int *v = NULL;      /* input array */
    int must_be_valid = 1;
    int valid;          /* the result computed by this program (1 if the array contains {0, 1, ... n-1}, 0 otherwise) */

    MPI_Init( &argc, &argv );
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1)
        n = atoi(argv[1]);

    if (argc > 2)
        must_be_valid = atoi(argv[2]);

    if (my_rank == 0) {
        if ((n % comm_sz) != 0) {
            fprintf(stderr, "FATAL: array size (%d) must be a multiple of %d\n", n, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        /* The master initializes `v[]` */
        v = (int*)malloc(n * sizeof(*v)); assert(v != NULL);
        fill(v, n, must_be_valid);
    }

#ifdef SERIAL
    if (my_rank == 0) {
        valid = 1;
        for (int i=0; i<n && valid; i++) {
            valid = (v[i] == i);
        }
    }
#else
    const int local_size = n / comm_sz;
    int *local_v = (int*)malloc( local_size * sizeof(*local_v) );
    assert(local_v != NULL);

    /**
     ** Distribute v[] across the nodes
     **/
    MPI_Scatter(v,              /* sendbuf      */
                local_size,     /* sendcount    */
                MPI_INT,        /* sendtype     */
                local_v,        /* recvbuf      */
                local_size,     /* recvcount    */
                MPI_INT,        /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD  /* comm         */
                );

    /**
     ** Each node checks its local chunk; note that it is important to
     ** handle indexes correctly!
     **/
    const int local_start = my_rank * (n / comm_sz);
    int local_valid = 1;
    for (int local_i = 0; local_i < local_size && local_valid; local_i++) {
        local_valid = (local_v[local_i] == local_start + local_i);
    }

    /**
     ** Rank 0 performs a logical and of all local_valid; the result
     ** is 1 iff all chunks are valid, 0 otherwise. Note that no
     ** synchronization (barrier) is necessary before this operation.
     **/
    MPI_Reduce(&local_valid,    /* sendbuf      */
               &valid,          /* recvbuf      */
               1,               /* count        */
               MPI_INT,         /* type         */
               MPI_LAND,        /* logical AND  */
               0,               /* root         */
               MPI_COMM_WORLD   /* comm         */
               );
    free(local_v);
#endif

    if (my_rank == 0) {
        printf("The sequence ");
        if (valid) {
            printf("is ");
        } else {
            printf("is not ");
        }
        printf("valid\n");
        free(v);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
