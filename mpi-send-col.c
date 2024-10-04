/****************************************************************************
 *
 * mpi-send-col.c - MPI Datatypes
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
% HPC - MPI Datatypes
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-11-18

When you need to decompose a rectangular domain of size $R \times C$
among $P$ MPI processes, you usually employ a block decomposition by
rows (Block, \*): the first process handles the first $R / P$ lines;
the second handles the next $R / P$ lines, and so on. Indeed, in the C
language matrices are stored in row-major order, and a (Block, \*)
decomposition has each partition as a contiguous sub-array so that
data transfers are greatly simplified.

In this exercise, on the other hand, we consider a column-wise
decomposition in order to use MPI derived datatypes.

Write a program where processes 0 and 1 keep a local matrix of size
$\textit{SIZE} \times (\textit{SIZE}+2)$ that include two columns of
_ghost cells_ (also called _halo_). In
[mpi-send-col.c](mpi-send-col.c) we set _SIZE_=4, but the program must
work with any value.

Process 0 and 1 initialize their local matrices as follows.

![](mpi-send-col1.svg)

Process 0 must send the _rightmost_ column to process 1, where it is
inserted into the _leftmost_ ghost area:

![](mpi-send-col2.svg)

Similarly, process 1 must send the _leftmost_ column to process 0,
where it is placed into the _rightmost_ ghost area.

![](mpi-send-col3.svg)

You should define a suitable datatype to represent a matrix column,
and use two `MPI_Sendrecv()` operations to exchange the data. Note
that `MPI_Sendrecv()` is a collective communication operation, so that
_all_ processes are supposed to execute it. To simplify the code, it
is preferable to perform the data exchange as follows: first, each
process sends the _rightmost_ column to the partner, that receives it
on the _leftmost_ ghost area:

![](mpi-send-col4.svg)

Then, each process sends the _leftmost_ column to the partner, that
receives it on the _rightmost_ ghost area:

![](mpi-send-col5.svg)

Note that, by doing this, each process sends and receives the same
data to/from the same (local) memory location.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-send-col.c -o mpi-send-col.c

To execute:

        mpirun -n 2 ./mpi-send-col.c

## Files

- [mpi-send-col.c](mpi-send-col.c)

***/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Matrix size */
#define SIZE 4

/* Initialize matrix m with the values k, k+1, k+2, ..., from left to
   right, top to bottom. m must point to an already allocated block of
   (size+2)*size integers. The first and last column of m is the halo,
   which is set to -1. */
void init_matrix( int *m, int size, int k )
{
    int i, j;
    for (i=0; i<size; i++) {
        for (j=0; j<size+2; j++) {
            if ( j==0 || j==size+1) {
                m[i*(size+2)+j] = -1;
            } else {
                m[i*(size+2)+j] = k;
                k++;
            }
        }
    }
}

void print_matrix( int *m, int size )
{
    int i, j;
    for (i=0; i<size; i++) {
        for (j=0; j<size+2; j++) {
            printf("%3d ", m[i*(size+2)+j]);
        }
        printf("\n");
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    int my_mat[SIZE][SIZE+2];

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( 0 == my_rank && 2 != comm_sz ) {
        fprintf(stderr, "You must execute exactly 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
#ifndef SERIAL
    MPI_Datatype t_column;
    MPI_Type_vector( SIZE,      /* count        */
                     1,         /* blocklen     */
                     SIZE+2,    /* stride       */
                     MPI_INT,   /* datatype     */
                     &t_column );

    MPI_Type_commit(&t_column);
#endif
    if ( 0 == my_rank ) {
        init_matrix(&my_mat[0][0], SIZE, 0);
    } else if (1 == my_rank) {
        init_matrix(&my_mat[0][0], SIZE, SIZE*SIZE);
    }

#ifdef SERIAL
    /* [TODO] Exchange borders here */
#else
    int other = 1 - my_rank;

    /* Send right column to neighbor */
    MPI_Sendrecv( &my_mat[0][SIZE],     /* sendbuf      */
                  1,                    /* sendcount    */
                  t_column,             /* datatype     */
                  other,                /* dest         */
                  0,                    /* sendtag      */
                  &my_mat[0][0],        /* recvbuf      */
                  1,                    /* recvcount    */
                  t_column,             /* datatype     */
                  other,                /* source       */
                  0,                    /* recvtag      */
                  MPI_COMM_WORLD,
                  MPI_STATUS_IGNORE
                  );

    /* Send left column to neighbor */
    MPI_Sendrecv( &my_mat[0][1],        /* sendbuf      */
                  1,                    /* sendcount    */
                  t_column,             /* datatype     */
                  other,                /* dest         */
                  0,                    /* sendtag      */
                  &my_mat[0][SIZE+1],   /* recvbuf      */
                  1,                    /* recvcount    */
                  t_column,             /* datatype     */
                  other,                /* source       */
                  0,                    /* recvtag      */
                  MPI_COMM_WORLD,
                  MPI_STATUS_IGNORE
                  );
#endif

    /* Print the matrices after the exchange; to do so without
       interference we must use this funny strategy: process 0 prints,
       then the processes synchronize, then process 1 prints. */
    if ( 0 == my_rank ) {
        printf("\n\nProcess 0:\n");
        print_matrix(&my_mat[0][0], SIZE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if ( 1 == my_rank ) {
        printf("\n\nProcess 1:\n");
        print_matrix(&my_mat[0][0], SIZE);
    }

#ifndef SERIAL
    MPI_Type_free(&t_column);
#endif
    MPI_Finalize();
    return EXIT_SUCCESS;
}
