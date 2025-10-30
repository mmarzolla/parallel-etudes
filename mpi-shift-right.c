/****************************************************************************
 *
 * mpi-shift-right.c - Circular shift of an array
 *
 * Copyright (C) 2022 Moreno Marzolla
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
% Circular shift of an array
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-10-30

Write an MPI program that performs a _right circular shift_ of an
array `v[N]`. Specifically, given an array $v = [v_0, v_1, \ldots,
v_{N-1}]$, its right circular shift is the array $v' = [v_{N-1}, v_0,
v_1, \ldots, v_{N-2}]$. In other words, each element of $v$ is moved
one position to the right, and the last element of $v$ becomes the
first element.

Assume that the length $N$ of $v$ is an integer multiple of the number
$P$ of MPI processes.

Compile with:

        mpicc -std=c99 -Wall -Wpedantic mpi-rotate-right.c -o mpi-rotate-right

Execute with:

        mpirun -n 4 ./mpi-rotate-right [N]

## File

- [mpi-shift-right.c](mpi-shift-right.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, N;
    int *v = NULL;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        N = atoi(argv[1]);
    } else {
        N = comm_sz * 10;
    }

    if ((N % comm_sz != 0) && (my_rank == 0)) {
        fprintf(stderr, "FATAL: array length (%d) must be a multiple of comm_sz (%d)\n", N, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* The master initializes the input array. */
    if ( 0 == my_rank ) {
        v = (int*)malloc(N * sizeof(*v));
        assert(v != NULL);
        printf("Before: [");
        for (int i=0; i<N; i++) {
            v[i] = i+1;
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

#ifdef SERIAL
    /* TODO: this is not a true parallel version, since the master
       does everything. */
    if ( 0 == my_rank ) {
        const int last_v = v[N-1];
        for (int i=N-2; i>=0; i--) {
            v[i+1] = v[i];
        }
        v[0] = last_v;
    }
#else

    const int prev = my_rank > 0 ? my_rank - 1 : comm_sz - 1;
    const int succ = my_rank < comm_sz-1 ? my_rank + 1 : 0;

    /* All processes initialize the local buffer. */
    const int local_N = N / comm_sz;
    int *local_v = (int*)malloc(local_N * sizeof(*local_v));
    assert(local_v != NULL);

    /* The master distributes the array `v[]`. */
    MPI_Scatter( v,             /* senfbuf      */
                 local_N,       /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 local_v,       /* recvbuf      */
                 local_N,       /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );

    /* Each process performs a local rotation; before the rotation, it
       is necessary to save the last element of `local_v[]`, since it
       needs to be sent to the next process. */
    const int last = local_v[local_N - 1];
    for (int i = local_N-1; i > 0; i--) {
        local_v[i] = local_v[i-1];
    }

    /* Each process sends `last` to the next process. Care must be
       taken here, since this involves both a send and receive
       operation and may produce a deadlock is not done correctly.
       The preferred way to do so is through the `MPI_Sendrecv()`
       primitive, that is guaranteed to be deadlock-free. */
    MPI_Sendrecv(&last,         /* sendbuf      */
                 1,             /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 succ,          /* dest         */
                 0,             /* sendtag      */
                 &local_v[0],   /* recvbuf      */
                 1,             /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 prev,          /* source       */
                 0,             /* recvtag      */
                 MPI_COMM_WORLD, /* comm        */
                 MPI_STATUS_IGNORE /* status    */
                 );

    /* The master assembles the local arrays. */
    MPI_Gather(local_v,         /* sendbuf      */
               local_N,         /* sendcount    */
               MPI_INT,         /* sendtype     */
               v,               /* recvbuf      */
               local_N,         /* recvcount    */
               MPI_INT,         /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* comm         */
               );

    free(local_v);

#endif
    if ( 0 == my_rank ) {
        printf("After: [");
        for (int i=0; i<N; i++) {
            printf("%d ", v[i]);
        }
        printf("]\n");
    }

    free(v);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
