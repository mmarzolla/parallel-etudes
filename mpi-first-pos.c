/****************************************************************************
 *
 * mpi-first-pos.c - First occurrence of a value in a vector
 *
 * Copyright (C) 2022--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - First occurrence of a value in a vector
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last updated: 2024-10-05

Write a MPI program that solves the following problem. Given a
non-empty integer array `v[0..N-1]` of length $N$, and an integer
value $k$, find the position (index) of the first occurrence of $k$ in
`v[]`; if $k$ is not present, the result will be $N$ (which is not a
valid index of the array, so we know that $N$ is not a valid result).

For example, if `v[] = {3, 15, -1, 15, 21, 15, 7}` and `k = 15`, the
result is 1, since `v[1]` is the first occurrence of `15`. If $k$ were
37, the result is 7, since 37 is not present and the length of the
array must be returned.

You may assume that:

- the array length $N$ is much larger than the number of MPI processes $P$.

- The array length $N$ is an integer multiple of $P$.

- At the beginning, the array length $N$, the value of $k$ and the
  content of `v[]` are all known by process 0 only.

- At the end, process 0 should receive the result.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-first-pos.c -o mpi-first-pos

To execute:

        mpirun -n 4 ./mpi-first-pos [N [k]]

## Files

- [mpi-first-pos.c](mpi-first-pos.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

/**
 * Returns a random value in [a, b]
 */
int randab(int a, int b)
{
    return a + rand() % (b - a + 1);
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, N, k, pos, minpos;
    int *v = NULL;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    } else {
        N = comm_sz * 10;
    }

    if ( argc > 2 ) {
        k = atoi(argv[2]);
    } else {
        k = randab(-1, N-1);
    }

    if ((N % comm_sz != 0) && (my_rank == 0)) {
        fprintf(stderr, "FATAL: array length (%d) must be a multiple of comm_sz (%d)\n", N, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* The array is initialized by process 0 */
    if ( 0 == my_rank ) {
        v = (int*)malloc(N * sizeof(*v));
        assert(v != NULL);
        printf("Before: [");
        for (int i=0; i<N; i++) {
            v[i] = i;
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

    /* This implementation receives the values `N` and `k` from the
       command line. These value could therefore be accessed by all
       processes. However, the specification asserts that only process
       0 knows then, so we need to broadcast the values to all other
       processes. */
    MPI_Bcast(&N,       /* buffer       */
              1,        /* count        */
              MPI_INT,  /* datatype     */
              0,        /* root         */
              MPI_COMM_WORLD );

    MPI_Bcast(&k,       /* buffer       */
              1,        /* count        */
              MPI_INT,  /* datatype     */
              0,        /* root         */
              MPI_COMM_WORLD );

    /* All processes initialize the local buffers */
    const int local_N = N / comm_sz;
    int *local_v = (int*)malloc(local_N * sizeof(*local_v));
    assert(local_v != NULL);

    /* The master distributes `v[]` to the other processes */
    MPI_Scatter( v,             /* senfbuf      */
                 local_N,       /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 local_v,       /* recvbuf      */
                 local_N,       /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );

    /* Every process performs a local sequential search on the local
       portion of `v[]`. There are two problems: (i) all local indices
       must be mapped to the corresponding global indices; (ii) since
       the result is computed as the min-reduction of the partial
       results, if a process does not find the key on the local array,
       it must send `N` to process 0. */
    int i = 0;
    while (i<local_N && local_v[i] != k) {
        i++;
    }
    if (i<local_N) {
        pos = my_rank * local_N + i; /* map local indices to global indices */
    } else {
        pos = N;
    }

    /* Performs a min-reduction of the local results */
    MPI_Reduce(&pos,    /* sendbuf      */
               &minpos, /* recvbuf      */
               1,       /* count        */
               MPI_INT, /* datatype     */
               MPI_MIN, /* op           */
               0,       /* root         */
               MPI_COMM_WORLD );

    if ( 0 == my_rank ) {
        printf("Result: %d\n", minpos);
    }

    free(v);
    free(local_v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
