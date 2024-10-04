/****************************************************************************
 *
 * mpi-ring.c - Ring communication with MPI
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <moreno.marzolla@unibo.it>
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
% HPC - Ring communication with MPI
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-06

Write a MPI program [mpi-ring.c](mpi-ring.c) that implements a message
exchange along a ring. Let $P$ be the number of MPI processes; the
program should behave according to the following specification:

- The program receives an integer $K \geq 1$ from the command
  line. $K$ is the number of "turns" of the ring. Since all MPI
  processes have access to the command line parameters, they know the
  value $K$ without the need to communicate.

- Process 0 (the master) sends process 1 an integer, whose
  initial value is 1.

- Every process $p$ (including the master) receives a value $v$ from
  the predecessor $p-1$, and sends $(v + 1)$ to the successor
  $p+1$. The predecessor of 0 is $(P - 1)$, and the successor of $(P -
  1)$ is 0.

- The master prints the value received after the $K$-th iteration and
  the program terminates. Given the number $P$ of processes and the
  value of $K$, what value should be printed by the master?

For example, if $K = 2$ and there are $P = 4$ processes, the
communication should be as shown in Figure 1 (arrows are messages
whose content is the number reported above them). There are $K = 2$
turns of the ring; at the end, process 0 receives and prints 8.

![Figure 1: Ring communication](mpi-ring.svg)

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-ring.c -o mpi-ring

To execute:

        mpirun -n 4 ./mpi-ring

## Files

- [mpi-ring.c](mpi-ring.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, K = 10;
#ifndef SERIAL
    int val;
#endif

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        K = atoi(argv[1]);
    }
#ifdef SERIAL
    /* [TODO] Rest of the code here... */
#else
    const int prev = (my_rank - 1 + comm_sz) % comm_sz;
    const int next = (my_rank + 1) % comm_sz;

    if ( 0 == my_rank ) {
        val = 1;
        MPI_Send(&val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    for (int k=0; k<K; k++) {
        MPI_Recv( &val,                 /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  prev,                 /* source       */
                  MPI_ANY_TAG,          /* tag          */
                  MPI_COMM_WORLD,       /* communicator */
                  MPI_STATUS_IGNORE     /* status       */
                  );
        if ( 0 != my_rank || k < K-1 ) {
            val++;
            MPI_Send( &val,             /* buf          */
                      1,                /* count        */
                      MPI_INT,          /* datatype     */
                      next,             /* dest         */
                      0,                /* tag          */
                      MPI_COMM_WORLD    /* communicator */
                      );
        }
    }

    if ( 0 == my_rank ) {
        const int expected = comm_sz * K;
        printf("expected=%d, received=%d\n", expected, val);
        if ( expected == val ) {
            printf("Test OK\n");
        } else {
            printf("Test FAILED: expected value %d at rank 0, got %d\n", expected, val);
        }
    }
#endif

    MPI_Finalize();
    return EXIT_SUCCESS;
}
