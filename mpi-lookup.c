/****************************************************************************
 *
 * mpi-lookup.c - Parallel linear search
 *
 * Copyright (C) 2021--2025 Moreno Marzolla
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
% Parallel linear search
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-01-13

Write an MPI program that finds the positions of all occurrences of a
given key `key` in an unsorted integer array `v[]`. For example, if
`v[] = {1, 3, -2, 3, 4, 3, 3, 5, -10}` and `key = 3`, the program must
build the result array

        {1, 3, 5, 6}

whose elements correspond to the positions (indices) of `key` in
`v[]`. Assume that:

- The array `v[]` is initially known by process 0 only.

- The length of `v[]` is an integer multiple of the number of
  processes.

- All processes know the value `key`, which is a compile-time
  constant.

- At the end, the result array must reside in the local memory of
  process 0.

![Figure 1: Communication scheme.](mpi-lookup.svg)

The program should operate as shown in Figure 1; `comm_sz` is the
number of MPI processes, and `my_rank` the rank of the process running
the code:

1. The master distributed `v[]` evenly across the processes. Assume
   that `n` is an exact multiple of `comm_sz`. Each process stores the
   local chunk in the `local_v[]` array of size `n / comm_sz`.

2. Each process computes the number `local_nf` of occurrences of `key`
   within `local_v[]`.

3. Each process creates a local array `local_result[]` of length
   `local_nf`, and fills it with the indices of the occurrences of
   `key` in `local_v[]`. **Warning**: indexes must refer to the
   **global** array `v[]` array, not `local_v[]`.

4. All processes use `MPI_Gather()` to concatenate the values of
   `local_nf` into an array `recvcounts[]` of length `comm_sz` owned
   by process 0.

5. Process 0 computes the _exclusive scane_ of `recvcounts[]`, and
   stores the result in a separate array `displs[]`. Only the master
   needs to know the content of `displs[]`, that is used at the next
   step.

6. All processes use `MPI_Gatherv()` to concatenate the arrays
   `local_result[]` to process 0. Process 0 uses the `displs[]` array
   from the previous step; all other processes do not need the
   displacements array, so they can pass a `NULL` reference.

> **Note:** steps 4 and 5 could be collapsed and realized more
> efficiently using `MPI_Exscan()` (not discussed during the class)
> that performs an exclusive scan.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-lookup.c -o mpi-lookup -lm

To execute:

        mpirun -n P ./mpi-lookup [N]

Example:

        mpirun -n 4 ./mpi-lookup

## Files

- [mpi-lookup.c](mpi-lookup.c)

***/
#include <stdio.h>
#include <stdlib.h> /* for rand() */
#include <time.h>   /* for time() */
#include <assert.h>
#include <mpi.h>

void fill(int *v, int n)
{
    for (int i=0; i<n; i++) {
        v[i] = (rand() % 100);
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int n = 1000;       /* input length */
    int *v = NULL;      /* input array */
    int *result = NULL; /* array of index of occurrences */
    int nf = 0;         /* number of occurrences */
    const int KEY = 42; /* lookup key */

    MPI_Init( &argc, &argv );
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc > 1)
        n = atoi(argv[1]);

    if (my_rank == 0) {
        if ((n % comm_sz) != 0) {
            fprintf(stderr, "FATAL: array size (%d) must be a multiple of %d\n", n, comm_sz);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        /* The master initializes `v[]` */
        v = (int*)malloc(n * sizeof(*v)); assert(v != NULL);
        fill(v, n);
    }

#ifdef SERIAL
    if (my_rank == 0) {

        /* Count the number of occurrences of `KEY` in `v[]` */
        nf = 0;
        for (int i=0; i<n; i++) {
            if (v[i] == KEY)
                nf++;
        }

        /* allocate the result array */
        result = (int*)malloc(nf * sizeof(*result)); assert(result != NULL);

        /* fill the result array  */
        for (int r=0, i=0; i<n; i++) {
            if (v[i] == KEY) {
                result[r] = i;
                r++;
            }
        }
    }
#endif

    int *local_v = NULL;        /* local portion of `v[]` */
    int local_nf = 0;           /* n. of occurrences of `KEY` in `local_v[]` */
    int *displs = NULL;         /* `displs[]` used by `MPI_Gatherv()` */
    int *recvcounts = NULL;     /* `recvcounts[]` used by `MPI_Gatherv()` */
    int *local_result = NULL;   /* array of positions of `KEY` in `local_v[]` */

    /**
     ** Step 1: distribute `v[]` across all MPI processes
     **/
#ifdef SERIAL
    /*
    const int local_size = ... ;
    local_v = ... ;
    MPI_Scatter(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                secfcount,
                recvtype,
                root,
                MPI_COMM_WORLD);
    */
#else
    const int local_size = n / comm_sz;
    local_v = (int*)malloc( local_size * sizeof(*local_v) );
    assert(local_v != NULL);

    MPI_Scatter(v,              /* sendbuf      */
                local_size,     /* sendcount    */
                MPI_INT,        /* sendtype     */
                local_v,        /* recvbuf      */
                local_size,     /* recvcount    */
                MPI_INT,        /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD  /* comm         */
                );
#endif

    /**
     ** Step 2: each process computes the number of occurrences
     ** `local_nf` of `KEY` in `local_v[]`
     **/
#ifdef SERIAL
    /* [TODO] */
#else
    for (int i=0; i<local_size; i++) {
        if (local_v[i] == KEY)
            local_nf++;
    }
#endif

    /**
     ** Step 3: each process allocates an array `local_resul[]` where
     ** the positions (indexes) of `KEY` in `local_v[]` are stored.
     ** It is essential that the positions refer to `v[]`, not to
     ** `local_v[]`.
     **/
#ifdef SERIAL
    /* [TODO] */

    /*
      local_result = (int*)malloc( local_nf * sizeof(*local_result) );
      assert(local_result != NULL);

      Fill local_result[] here appropriately...
    */
#else
    local_result = (int*)malloc( local_nf * sizeof(*local_result) );
    assert(local_result != NULL);

    for (int r=0, i=0; i<local_size; i++) {
        if (local_v[i] == KEY) {
            local_result[r] = my_rank * local_size + i; /* the indexes refer to `v[]` */
            r++;
        }
    }
#endif

    /**
     ** Step 4: Process 0 gathers all values `local_nf` into a local
     ** array `recvcounts[]` of size `comm_sz`
     **/
#ifdef SERIAL
    /*
    if (my_rank == 0) {
        displs = (int*)malloc( comm_sz * sizeof(*displs) ); assert(displs != NULL);
        recvcounts = (int*)malloc( comm_sz * sizeof(*recvcounts) ); assert(recvcounts != NULL);
    }

    MPI_Gather(sendbuf,
               sendcount,
               sendtype,
               recvbuf,
               recvcount,
               recvtype,
               root,
               MPI_COMM_WORLD);
    */
#else
    if (my_rank == 0) {
        displs = (int*)malloc( comm_sz * sizeof(*displs) ); assert(displs != NULL);
        recvcounts = (int*)malloc( comm_sz * sizeof(*recvcounts) ); assert(recvcounts != NULL);
    }

    MPI_Gather(&local_nf,       /* sendbuf      */
               1,               /* sendcount    */
               MPI_INT,         /* sendtype     */
               recvcounts,      /* recvbuf      */
               1,               /* recvcount    */
               MPI_INT,         /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* comm         */
               );
#endif

    /**
     ** Step 5: process 0 performs an exclusive scan of `recvcounts[]`
     ** and stores the result into a new array `displs[]`. Then,
     ** another array `result[]` is created, big enough to store all
     ** occurrences received from all processes.
     **/
    if (my_rank == 0) {
#ifdef SERIAL
        /* [TODO] */
#else
        displs[0] = 0;
        nf = recvcounts[0];
        for (int i=1; i<comm_sz; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
            nf += recvcounts[i];
        }
        result = (int*)malloc(nf * sizeof(*result));
        assert(result != NULL);
#endif
    }

    /**
     ** Step 6: process 0 gathers `local_result[]` into `result[]`
     **/
#ifdef SERIAL
    /*
    MPI_Gatherv(sendbuf,
                sendcount,
                sendtype,
                recvbuf,
                recvcounts,
                displacements,
                recvtype,
                root,
                MPI_COMM_WORLD);
    */
#else
    MPI_Gatherv(local_result,   /* sendbuf      */
                local_nf,       /* sendcount    */
                MPI_INT,        /* sendtype     */
                result,         /* recvbuf      */
                recvcounts,     /* recvcounts   */
                displs,         /* displacements */
                MPI_INT,        /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD
                );
#endif

    free(displs);
    free(recvcounts);
    free(local_v);
    free(local_result);

    if (my_rank == 0) {
        printf("There are %d occurrences of %d\n", nf, KEY);
        printf("Positions: ");
        for (int i=0; i<nf; i++) {
            printf("%d ", result[i]);
            if (v[result[i]] != KEY) {
                fprintf(stderr, "\nFATAL: v[%d]=%d, expected %d\n", result[i], v[result[i]], KEY);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        printf("\n");
        free(v);
        free(result);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
