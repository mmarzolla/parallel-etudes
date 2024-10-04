/****************************************************************************
 *
 * mpi-my-reduce.c - Sum-reduce using point-to-point communications
 *
 * Copyright (C) 2013 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Sum-reduce using point-to-point communication
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
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
