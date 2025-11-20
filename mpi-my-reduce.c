/****************************************************************************
 *
 * mpi-my-reduce.c - Sum-reduction using point-to-point communications
 *
 * Copyright (C) 2025 Moreno Marzolla
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
% Sum-reduce using point-to-point communication
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2025-11-20

Write a function `my_reduce()` with the following prototype:

```C
int my_Reduce(const int *v)
```

that computes the sum-reduction of integers stored in the local memory
of each MPI process. Specifically, the function has the following
semantics:

- All MPI processes call `my_reduce()` passing a pointer `v` to a
  single integer value stored in their local memory space.

- For rank 0: the function returns the sum-reduction of all `*v`; for
  the other ranks, the return value is arbitrary.

You are allowed to use only point-to-point communication primitives,
i.e., `MPI_Send()` / `MPI_Recv()` and/or their asynchronous versions.

# Suggestion

A simple, but not efficient, solution would be for all processes to
send `*v` to the master using `MPI_Send()`. A better way would be to
structure the communication as a tree, i.e., rank $i$ receives
messages from ranks $(2i+1)$ and $(2i+2)$, and send the sum to rank
$\lceil i/2 \rceil$. See Figure 1.

![Figure 1: Communicaton pattern for the `my_Reduce()` function.](mpi-my-reduce.svg)

# Files

- [mpi-my-reduce.c](mpi-my-reduce.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int my_Reduce(const int *v)
{
#ifdef SERIAL
    /* TODO */
    return 0;
#else
    int my_rank, comm_sz;
    int result = *v, buf;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const int left_child = (2*my_rank + 1 < comm_sz ? 2*my_rank + 1 : MPI_PROC_NULL);
    const int right_child = (2*my_rank + 2 < comm_sz ? 2*my_rank + 2 : MPI_PROC_NULL);
    const int parent = (my_rank > 0 ? (my_rank - 1)/2 : MPI_PROC_NULL);

    buf = 0;
    MPI_Recv( &buf,                 /* buf          */
              1,                    /* count        */
              MPI_INT,              /* datatype     */
              left_child,           /* source       */
              0,                    /* tag          */
              MPI_COMM_WORLD,       /* comm         */
              MPI_STATUS_IGNORE     /* status       */
              );
    result += buf;

    buf = 0;
    MPI_Recv( &buf,                 /* buf          */
              1,                    /* count        */
              MPI_INT,              /* datatype     */
              right_child,          /* source       */
              0,                    /* tag          */
              MPI_COMM_WORLD,       /* comm         */
              MPI_STATUS_IGNORE     /* status       */
              );
    result += buf;

    MPI_Send( &result,              /* buf          */
              1,                    /* count        */
              MPI_INT,              /* datatype     */
              parent,               /* destination  */
              0,                    /* tag          */
              MPI_COMM_WORLD        /* comm         */
              );

    return result;
#endif
}


int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    v = my_rank;

    const int result = my_Reduce(&v);
    const int expected = (comm_sz * (comm_sz-1)) / 2;

    if ( 0 == my_rank) {
        if (result == expected)
            printf("OK: received %d\n", result);
        else
            printf("FAILED: received %d, expected %d\n", result, expected);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
