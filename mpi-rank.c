/****************************************************************************
 *
 * mpi-rank.c - Rank elements of an array.
 *
 * Copyright (C) 2026 Moreno Marzolla
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
% Rank elements of an array
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-06-29

Given an array $v$ of length $n$, the _rank_ $r[i]$ of $v[i]$ if the
number of elements that are lower than $v[i]$ (this definition assumes
that $v$ does not contain duplicate values; however, it can be
extended to the general case, as is done by this program).  $r[i]$ is
the position that $v[i]$ would occupy if the array $v$ were sorted;
therefore, the array of ranks defines a sorting permutation of $v$.

From the discussion above, it is possible to compute the ranks by
simply sorting $v$ and keeping track of the sorting permutation. This
can be accomplished in $\Theta(n \log n)$ serial time using an
efficient general-purpose sorting algorithm.

The goal of this exercise is to write a distributed-memory version of
the trivial ranking algorithm that works by comparing each element
$v[i]$ with all other elements, and count how many of them are lower
than $v[i]$. The algorithm assumes that $v$ does not contain duplicate
values.

![Figure 1: MPI rank.](mpi-rank.svg)

To this aim, each MPI process keeps two local arrays: `local_v[]` that
contains a chunk of `v`, and `received_v[]` that contains a (possibly)
different one. proceed as follows (see Figure 1):

1. Split $v$ across all MPI processes using `MPI_Scatter()`, each
   process sets `received_v[]` to the same content of `local_v[]`.

2. Each process updates local ranks by comparing all elements in
   `local_v[]` with all elements in `received_v[]`.

3. Each process sends `received_v[]` to its successor using
   `MPI_Sendrecv()` to avoid possible deadlocks. Go back to step 2.

If there are $P$ processes, step 3 needs to be repeated $P-1$ times.
In this way, each process can compare its own elements with all other
ones.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-rank.c -o mpi-rank

To execute:

        mpirun -n P ./mpi-rank

## Files

- [mpi-rank.c](mpi-rank.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>

/* Update the `local_rank` array, by comparing all values in `local_v`
   to all values in `received_v`. */
void update_ranks(const int *local_v, const int *received_v, int *local_rank, int local_n)
{
    for (int i=0; i<local_n; i++) {
        for (int j=0; j<local_n; j++) {
            if (local_v[i] > received_v[j])
                local_rank[i]++;
        }
    }
}

#ifndef SERIAL
/* Send `received_v` to neighbor, and receive from neighbor. It is
   possible to use MPI_Sendrecv_replace() to do the same thing without
   the need of an additional buffer `tmp`. */
void circulate_received_v(int *received_v, int local_n, int pred, int succ)
{
    int *tmp = (int*)malloc(local_n * sizeof(*tmp)); assert(tmp != NULL);

    MPI_Sendrecv(received_v,    /* sendbuf      */
                 local_n,       /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 succ,          /* dest         */
                 0,             /* sendtag      */
                 tmp,           /* recvbuf      */
                 local_n,       /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 pred,          /* source       */
                 MPI_ANY_TAG,   /* recvtag      */
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    memcpy(received_v, tmp, local_n * sizeof(*tmp));
    free(tmp);
}
#endif

int main( int argc, char *argv[])
{
    int *v = NULL, *rank = NULL;
#ifndef SERIAL
    int  *local_v, *received_v, *local_rank;
#endif
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const int n = 1000 * comm_sz;

    if (my_rank == 0) {
        v = (int*)malloc(n * sizeof(*v));
        rank = (int*)malloc(n * sizeof(*v));
        for (int i=0; i<n; i++) {
            v[i] = i;
        }
    }

    assert(n % comm_sz == 0); /* requires that comm_sz divides n. */

#ifdef SERIAL
    /* [TODO] This is not a true parallel version, since process 0
       does everything. */
    if (0 == my_rank) {
        for (int i=0; i<n; i++) {
            rank[i] = 0;
        }
        update_ranks(v, v, rank, n);
    }
#else
    const int pred = (my_rank - 1 + comm_sz) % comm_sz;
    const int succ = (my_rank + 1) % comm_sz;
    const int local_n = n / comm_sz;

    /* Allocate local buffers. */
    local_v = (int*)malloc( local_n * sizeof(*local_v)); assert(local_v != NULL);
    received_v = (int*)malloc( local_n * sizeof(*received_v)); assert(received_v != NULL);
    local_rank = (int*)malloc( local_n * sizeof(*local_rank)); assert(local_rank != NULL);

    /* Distribute input array. */
    MPI_Scatter(v,              /* sendbuf      */
                local_n,        /* sendcount    */
                MPI_INT,        /* datatype     */
                local_v,        /* recvbuf      */
                local_n,        /* recvcount    */
                MPI_INT,        /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD);

    memcpy(received_v, local_v, local_n * sizeof(*local_v));

    /* Initialize local ranks to zero. */
    for (int i=0; i<local_n; i++) {
        local_rank[i] = 0;
    }

    for (int round=0; round < comm_sz; round++) {
        if (0 == my_rank) 
            printf("Round %d of %d\n", round+1, comm_sz);
        update_ranks(local_v, received_v, local_rank, local_n);

        if (round < comm_sz-1)
            circulate_received_v(received_v, local_n, pred, succ);
    }

    /* Concatenate local ranks. */
    MPI_Gather(local_rank,      /* sendbuf      */
               local_n,         /* sendcount    */
               MPI_INT,         /* sendtype     */
               rank,            /* recvbuf      */
               local_n,         /* recvcount    */
               MPI_INT,         /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD);
#endif
    /* Check results. */
    if ( 0 == my_rank) {
        for (int i=0; i<n; i++) {
            if (rank[i] != i) {
                printf("FATAL: rank[%d] == %d, expected %d\n", i, rank[i], i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        printf("Check OK\n");
    }

#ifndef SERIAL
    free(local_v);
    free(received_v);
    free(local_rank);
#endif
    free(v);
    free(rank);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
