/****************************************************************************
 *
 * mpi-gemv.c - Matrix-Vector multiply
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
% Matrix-Vector Multiply
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-01-24

Compute the matrix-vector product $Ab = c$ using MPI. $A$ is a square
matrix of size $n \times n$. You should assume that both $A$, $b$ and
the length $n$ are known to the root (rank 0) process only.  Also,
assume that $n$ is a multiple of the number $P$ of MPI processes.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-gemv.c -o mpi-gemv -lm

To execute:

        mpirun -n P ./mpi-gemv [n]

Example:

        mpirun -n 4 ./mpi-gemv 1000

## Files

- [mpi-gemv.c](mpi-gemv.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/**
 * Compute the matrix-vector product Ab = c. The parameters are
 * significant on the root only (rank=0); for all other ranks, they
 * are undefined.
 */
void gemv( float* A, float* b, float *c, int n )
{
#ifdef SERIAL
    for (int i=0; i<n; i++) {
        c[i] = 0;
        for (int j=0; j<n; j++) {
            c[i] += A[i*n + j] * b[j];
        }
    }
#else
    float *local_A, *local_c;
    int local_n;
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0 && (n % comm_sz)) {
        fprintf(stderr, "FATAL: n (%d) is not a multiple of comm_sz (%d)\n", n, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Bcast(&n,       /* buffer       */
              1,        /* count        */
              MPI_INT,  /* datatype     */
              0,        /* root         */
              MPI_COMM_WORLD
              );

    local_n = n / comm_sz;
    local_A = (float*)malloc(local_n * n * sizeof(*local_A));
    assert(local_A != NULL);
    /* vector b points to NULL for all processes except the root; we
       allocate b on all other nodes, now that we know its length
       n. */
    if (0 != my_rank) {
        b = (float*)malloc(n * sizeof(*b));
        assert(b);
    }
    local_c = (float*)malloc(local_n * sizeof(*local_A));
    assert(local_c != NULL);

    /* Distribute A */
    MPI_Scatter(A,              /* sendbuf      */
                local_n * n,    /* sendcount    */
                MPI_FLOAT,      /* sendtype     */
                local_A,        /* recvbuf      */
                local_n * n,    /* recvcount    */
                MPI_FLOAT,      /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD
                );

    /* Replicate b */
    MPI_Bcast(b,                /* buffer       */
              n,                /* count        */
              MPI_FLOAT,        /* datatype     */
              0,                /* root         */
              MPI_COMM_WORLD
              );

    /* Local computation */
    for (int i=0; i<local_n; i++) {
        local_c[i] = 0;
        for (int j=0; j<n; j++) {
            local_c[i] += local_A[i*n + j] * b[j];
        }
    }

    /* Gather result */
    MPI_Gather(local_c,         /* sendbuf      */
               local_n,         /* sendcount    */
               MPI_FLOAT,       /* sendtype     */
               c,               /* recvbuf      */
               local_n,         /* recvcount    */
               MPI_FLOAT,       /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD
               );

    free(local_A);
    if (0 != my_rank) free(b);
    free(local_c);
#endif
}

void init(float *A, float *b, int n)
{
    for (int i=0; i<n; i++) {
        b[i] = n;
        for (int j=0; j<n; j++) {
            A[i*n+j] = 1 / (float)n;
        }
    }
}

/**
 * Check result.
 */
void check_result(const float *c, int n)
{
    static const float TOL = 1e-5;
    for (int i=0; i<n; i++) {
        if (fabsf(c[i] - n) > TOL) {
            fprintf(stderr, "Check FAILED: c[%d]=%f, expected %f\n", i, c[i], (float)n);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    fprintf(stderr, "Check OK.\n");
}

int main( int argc, char* argv[] )
{
    float *A = NULL, *b = NULL, *c = NULL;
    int n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    if ( 0 == my_rank ) {
        /* The master allocates the data */
        A = (float*)malloc(n * n * sizeof(*A)); assert(A);
        b = (float*)malloc(n * sizeof(*b)); assert(b);
        c = (float*)malloc(n * sizeof(*c)); assert(c);
        init(A, b, n);
    }

    gemv(A, b, c, n);

    if (0 == my_rank)
        check_result(c, n);

    free(A);
    free(b);
    free(c);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
