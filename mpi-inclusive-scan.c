/****************************************************************************
 *
 * mpi-inclusive-scan.c - Inclusive Scan
 *
 * Copyright (C) 2016--2021 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

void fill(int* local, int n, int id)
{
    int i;

    for (i=0; i<n; i++) {
        local[i] = i + id * n;
    }
}

void clean(int* scan, int n)
{
    int i;

    for (i=0; i<n; i++) {
        scan[i] = 0;
    }
}

void sum(int* local, int* scan, int n, int id)
{
    int i;

    for (i=0; i<n; i++) {
        scan[i] = scan[i] + local[i];
    }
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *local_x, *scan_x;
    int len=3, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_sz);
    
    local_x = (int *)malloc(len*sizeof(*local_x));
    scan_x = (int *)malloc(len*sizeof(*scan_x));

    fill(local_x, len, my_rank);
    clean(scan_x, len);

#ifndef NAIVE

    if (my_rank > 0) {
        MPI_Recv( scan_x,           /* buf          */
                  len,              /* count        */
                  MPI_INT,          /* datatype     */
                  my_rank+1,        /* dest         */
                  my_rank,          /* tag          */
                  MPI_COMM_WORLD,   /* communicator */
                  MPI_STATUS_IGNORE /* status       */
                  );
    }

    sum(local_x, scan_x, len, my_rank);

    if (my_rank < comm_sz-1) {
        MPI_Send( scan_x,           /* buf          */
                  len,              /* count        */
                  MPI_INT,          /* datatype     */
                  my_rank+1,        /* dest         */
                  my_rank+1,        /* tag          */
                  MPI_COMM_WORLD    /* communicator */
                  );
    }

    for (i=0; i<len; i++) {
        printf("My rank: %d, scan[%d] = %d\n", my_rank, i, scan_x[i]);
    }

#else

    MPI_Scan( local_x,		        /* sendbuf      */
              scan_x,		        /* recvbuf      */
              comm_sz,		        /* count        */
              MPI_INT,		        /* datatype     */
              MPI_SUM,		        /* operator     */
              MPI_COMM_WORLD        /* communicator */
              );

#endif

    free(local_x);
    free(scan_x);

    MPI_Finalize();
    return EXIT_SUCCESS;
}