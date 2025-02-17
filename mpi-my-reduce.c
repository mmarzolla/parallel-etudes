/****************************************************************************
 *
 * mpi-my-reduce.c - Sum-reduce using point-to-point communications
 *
 * Copyright (C) 2013 Moreno Marzolla
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
% HPC - Sum-reduce using point-to-point communication
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2023-02-14

***/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int my_Reduce(const int *v)
{
    int my_rank, comm_sz;
    int result = *v, buf;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    const int src1 = 2*my_rank + 1; /* rank of "left" child in the communication tree */
    if (src1 < comm_sz) {
        MPI_Recv( &buf,                 /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  src1,                 /* source       */
                  0,                    /* tag          */
                  MPI_COMM_WORLD,       /* comm         */
                  MPI_STATUS_IGNORE     /* status       */
                  );
        result += buf;
    }
    const int src2 = 2*my_rank + 2; /* rank of "right" child in the communication tree */
    if (src2 < comm_sz) {
        MPI_Recv( &buf,                 /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  src2,                 /* source       */
                  0,                    /* tag          */
                  MPI_COMM_WORLD,       /* comm         */
                  MPI_STATUS_IGNORE     /* status       */
                  );
        result += buf;
    }

    if ( my_rank > 0 ) {
        const int dst = (my_rank - 1) / 2;
        MPI_Send( &result,              /* buf          */
                  1,                    /* count        */
                  MPI_INT,              /* datatype     */
                  dst,                  /* destination  */
                  0,                    /* tag          */
                  MPI_COMM_WORLD        /* comm         */
                  );
    }

    return result;
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
    const int expected = comm_sz * (comm_sz-1) / 2;

    if ( 0 == my_rank) {
        if (result == expected)
            printf("OK: received %d\n", result);
        else
            printf("FAILED: received %d, expected %d\n", result, expected);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
