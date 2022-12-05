/****************************************************************************
 *
 * mpi-my-bcast.c - Broadcast using point-to-point communications
 *
 * Copyright (C) 2017--2022 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Broadcast using point-to-point communication
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2022-10-24

The purpose of this exercise is to implement the function

        void my_Bcast(int *v)

which realizes a _broadcast_ communication, where the value `*v` that
resides on the local memory of process 0 is sent to all other
processes. In practice, this function should behave as

```C
MPI_Bcast(v,             \/\* buffer   \*\/
          1,             \/\* count    \*\/
          MPI_INT,       \/\* datatype \*\/
          0,             \/\* root     \*\/
          MPI_COMM_WORLD \/\* comm     \*\/
          );
```

> **Note**. `MPI_Bcast()` must always be preferred to any home-made
> solution. The purpose of this exercise is to learn how `MPI_Bcast()`
> might be implemented, although the actual implementations are
> architecture-dependent.

To implement `my_Bcast()`, each process determines its own rank $p$
and the number $P$ of active MPI processes.

Then, process 0:

- sends `*v` to processes $(2p + 1)$ and $(2p + 2)$, provided that
  they exist.

Any other process $p>0$:

- receives an integer from $(p - 1)/2$ and stores it in `*v`;

- sends `*v` to processes $(2p + 1)$ and $(2p + 2)$, provided that
  they exist.

For example, with $P = 15$ you get the communication scheme
illustrated in Figure 1; arrows indicate point-to-point
communications, numbers indicate the rank of processes. The procedure
described above works correctly for any $P$.

![Figure 1: Broadcast communication with $P = 15$ processes](mpi-my-bcast.svg)

The file [mpi-my-bcast.c](mpi-my-bcast.c) contains the skeleton of the
`my_Bcast()` function. Complete the implementation using
point-to-point send/receive operations.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-my-bcast.c -o mpi-my-bcast

To execute:

        mpirun -n 4 ./mpi-my-bcast

## Files

- [mpi-my-bcast.c](mpi-my-bcast.c)

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void my_Bcast(int *v)
{
    int my_rank, comm_sz;
#ifndef SERIAL
    int dest1, dest2;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
#ifdef SERIAL
    /* [TODO] */
#else
    if ( my_rank > 0 ) {
        MPI_Recv( v,                    /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  (my_rank-1)/2,        /* source       */
                  0,                    /* tag          */
                  MPI_COMM_WORLD,       /* comm         */
                  MPI_STATUS_IGNORE     /* status       */
                  );
    }
    dest1 = (2*my_rank + 1 < comm_sz ? 2*my_rank + 1 : MPI_PROC_NULL);
    dest2 = (2*my_rank + 2 < comm_sz ? 2*my_rank + 2 : MPI_PROC_NULL);
    /* sending a message using MPI_Send() to MPI_PROC_NULL has no
       effect (see man page for MPI_Send) */
    MPI_Send( v, 1, MPI_INT, dest1, 0, MPI_COMM_WORLD);
    MPI_Send( v, 1, MPI_INT, dest2, 0, MPI_COMM_WORLD);
#endif
}

#ifndef SERIAL
/**
 * Same as above, but using non-blocking send.
 */
void my_Ibcast(int *v)
{
    MPI_Request req[2];
    int my_rank, comm_sz, dest1, dest2;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    if ( my_rank > 0 ) {
        MPI_Recv( v,                    /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  (my_rank-1)/2,        /* source       */
                  0,                    /* tag          */
                  MPI_COMM_WORLD,       /* comm         */
                  MPI_STATUS_IGNORE     /* status       */
                  );
    }
    dest1 = (2*my_rank + 1 < comm_sz ? 2*my_rank + 1 : MPI_PROC_NULL);
    dest2 = (2*my_rank + 2 < comm_sz ? 2*my_rank + 2 : MPI_PROC_NULL);
    /* sending a message using MPI_Send() to MPI_PROC_NULL has no effect */
    MPI_Isend( v, 1, MPI_INT, dest1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Isend( v, 1, MPI_INT, dest2, 0, MPI_COMM_WORLD, &req[1]);
    /* Wait all pending requests to complete */
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
}
#endif

int main( int argc, char *argv[] )
{
    const int SENDVAL = 999; /* valore che viene inviato agli altri processi */
    int my_rank;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ( 0 == my_rank ) {
        v = SENDVAL; /* only process 0 sets the value to be sent */
    } else {
        v = -1; /* all other processes set v to -1; if everything goes well, the value will be overwritten with the value received from the master */
    }

    printf("Sending %d\n", SENDVAL);

    my_Bcast(&v);

    if ( v == 999 ) {
        printf("OK: ");
    } else {
        printf("ERROR: ");
    }
    printf("Process %d received %d\n", my_rank, v);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
