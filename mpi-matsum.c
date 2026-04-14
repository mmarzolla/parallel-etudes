/****************************************************************************
 *
 * mpi-matsum.c - Matrix sum
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
% Matrix Sum
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2026-04-14

Compute the matrix sum $A + B = C$ using MPI. $A, B, C$ are square
matrices of size $n \times n$. Assume that both $A, B$ and their size
$n$ are known to the root (rank 0) process only. Also, assume that $n$
is a multiple of the number $P$ of MPI processes.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-matsum.c -o mpi-matsum -lm

To execute:

        mpirun -n P ./mpi-matsum [n]

Example:

        mpirun -n 4 ./mpi-matsum 1000

## Files

- [mpi-gemv.c](mpi-gemv.c)

***/
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs() */
#include <assert.h>
#include <mpi.h>

/**
 * Compute the matrix-matrix sum A + B = C. The parameters are
 * significant on the root only (rank=0); for all other ranks, they
 * are undefined.
 *
 * Note that this implementation could be greatly simplified, since
 * matrices in C are stored as arrays, so this is really a
 * vector-vector sum.
 */
void matsum( const float* A, const float* B, float *C, int n )
{
#ifdef SERIAL
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            C[i*n + j] = A[i*n + j] + B[i*n + j];
        }
    }
#else
    float *local_A, *local_B, *local_C;
    int local_n;
    int my_rank, comm_sz;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0 && (n % comm_sz)) {
        fprintf(stderr, "FATAL: matrix side n (%d) is not a multiple of comm_sz (%d)\n", n, comm_sz);
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
    local_B = (float*)malloc(local_n * n * sizeof(*local_B));
    assert(local_B != NULL);
    local_C = (float*)malloc(local_n * n * sizeof(*local_C));
    assert(local_C != NULL);

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

    /* Distribute B */
    MPI_Scatter(B,              /* sendbuf      */
                local_n * n,    /* sendcount    */
                MPI_FLOAT,      /* sendtype     */
                local_B,        /* recvbuf      */
                local_n * n,    /* recvcount    */
                MPI_FLOAT,      /* recvtype     */
                0,              /* root         */
                MPI_COMM_WORLD
                );

    /* Local computation */
    for (int i=0; i<local_n; i++) {
        for (int j=0; j<n; j++) {
            local_C[i*n + j] = local_A[i*n + j] + local_B[i*n + j];
        }
    }

    /* Gather result */
    MPI_Gather(local_C,         /* sendbuf      */
               local_n * n,     /* sendcount    */
               MPI_FLOAT,       /* sendtype     */
               C,               /* recvbuf      */
               local_n * n,     /* recvcount    */
               MPI_FLOAT,       /* recvtype     */
               0,               /* root         */
               MPI_COMM_WORLD
               );

    free(local_A);
    free(local_B);
    free(local_C);
#endif
}

void init(float *A, float *B, int n)
{
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            A[i*n + j] = i+j;
            B[i*n + j] = n - A[i*n + j];
        }
    }
}

/**
 * Check result.
 */
void check_result(const float *C, int n)
{
    static const float TOL = 1e-5;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (fabsf(C[i*n + j] - n) > TOL) {
                fprintf(stderr, "Check FAILED: C[%d][%d]=%f, expected %f\n", i, j, C[i*n+j], (float)n);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }
    fprintf(stderr, "Check OK.\n");
}

int main( int argc, char* argv[] )
{
    float *A = NULL, *B = NULL, *C = NULL;
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
        A = (float*)malloc(n * n * sizeof(*A)); assert(A != NULL);
        B = (float*)malloc(n * n * sizeof(*B)); assert(B != NULL);
        C = (float*)malloc(n * n * sizeof(*C)); assert(C != NULL);
        init(A, B, n);
    }
    const float tstart = MPI_Wtime();
    matsum(A, B, C, n);
    const float elapsed = MPI_Wtime() - tstart;

    if (0 == my_rank) {
        printf("Elapsed time (s): %.3f\n", elapsed);
        check_result(C, n);
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
