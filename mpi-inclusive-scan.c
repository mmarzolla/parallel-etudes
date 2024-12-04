/*****************************************************************************
 *
 * mpi-inclusive-scan.c - Inclusive scan
 *
 * Copyright (C) 2023-2024 Moreno Marzolla <https://www.moreno.marzolla.name/>
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-inclusive-scan.c -o mpi-inclusive-scan
 *
 * Run with:
 * mpirun -n 4 ./mpi-inclusive-scan
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

void check(int *s, int n)
{
    for (int i = 0; i < n; i++) {
        if ( s[i] != i+1 ) {
            fprintf(stderr, "Check failed: expected s[%d]==%d, got %d\n", i, i+1, s[i]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    fprintf(stderr, "Check ok!\n");
}


void fill(int* local, int n)
{
    for (int i = 0; i < n; i++) {
        local[i] = 1;
    }
}

/* Compute the inclusive scan of the n-elements array v[], and store
   the result in s[]. v[] and s[] must not overlap */
void inclusive_scan(int *v, int *s, int n)
{
    s[0] = v[0];
    for (int i=1; i<n; i++) {
        s[i] = s[i-1] + v[i];
    }
}

/* Compute the exclusive scan of the n-elements array v[], and store
   the result in s[]. v[] and s[] must not overlap */
void exclusive_scan(int *v, int *s, int n)
{
    s[0] = 0;
    for (int i=1; i<n; i++) {
        s[i] = s[i-1] + v[i-1];
    }
}

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *in = NULL, *out = NULL;
    int *my_in, *my_out;
    int len = 100000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( 0 == my_rank ) {
        in = (int *)malloc(len * sizeof(*in)); assert(in != NULL);
        out = (int *)malloc(len * sizeof(*out)); assert(out != NULL);
        fill(in, len);
    }

    int sendcounts[comm_sz];
    int displs[comm_sz];
    for (int i=0; i<comm_sz; i++) {
        const int my_start = (len * i) / comm_sz;
        const int my_end = len * (i + 1) / comm_sz;
        sendcounts[i] = my_end - my_start;
        displs[i] = my_start;
    }

    const int my_len = sendcounts[my_rank];

    my_in = (int *)malloc(my_len * sizeof(*my_in)); assert(my_in != NULL);
    my_out = (int *)malloc(my_len * sizeof(*my_out)); assert(my_out != NULL);

    /* The master distributes the input array */

    MPI_Scatterv( in,           /* sendbuf      */
                  sendcounts,   /* sendcounts   */
                  displs,       /* displs       */
                  MPI_INT,      /* sendtype     */
                  my_in,        /* recvbuf      */
                  my_len,       /* recvcount    */
                  MPI_INT,      /* recvtype     */
                  0,            /* root         */
                  MPI_COMM_WORLD /* comm        */
                  );

    int blksum[comm_sz];
    int blksum_s[comm_sz];
    int my_blksum;

    /* Each process performs an inclusive scan of its portion of array */
    inclusive_scan(my_in, my_out, my_len);

    /* The master collects the last element of each local buffer `my_out[]` */
    MPI_Gather( &my_out[my_len - 1],    /* sendbuf      */
                1,                      /* sendcount    */
                MPI_INT,                /* sendtype     */
                blksum,                 /* recvbuf      */
                1,                      /* recvcount    */
                MPI_INT,                /* recvtype     */
                0,                      /* root         */
                MPI_COMM_WORLD          /* comm         */
                );

    /* The master performs an exclusive scan on blksum_s[] */
    if ( 0 == my_rank ) {
        exclusive_scan(blksum, blksum_s, comm_sz);
    }

    MPI_Scatter( blksum_s,      /* sendbuf      */
                 1,             /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 &my_blksum,    /* recvbuf      */
                 1,             /* recvcount    */
                 MPI_INT,       /* recvtype     */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );

    /* Each process increments all values of its portion of the array */
    for (int i=0; i<my_len; i++) {
        my_out[i] += my_blksum;
    }

    /* The master collects and concatenates the local arrays and
       checks the result. */
    MPI_Gatherv( my_out,        /* sendbuf      */
                 my_len,        /* sendcount    */
                 MPI_INT,       /* sendtype     */
                 out,           /* recvbuf      */
                 sendcounts,    /* recvcounts   */
                 displs,        /* displs       */
                 MPI_INT,       /* recvtype     */
                 0,             /* root         */
                 MPI_COMM_WORLD /* comm         */
                 );

    if ( 0 == my_rank) {
        check(out, len);
    }

    free(in);
    free(out);
    free(my_in);
    free(my_out);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
