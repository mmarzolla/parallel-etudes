/****************************************************************************
 *
 * mpi-gaussian-elimination.c - Solve systems of linear equations in upper triangular form
 *
 * Copyright (C) 2023 by Alice Girolomini <alice.girolomini(at)studio.unibo.it>
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
% HPC - Solution of a system of linear equations in upper triangular form
% Alice Girolomini <alice.girolomini@studio.unibo.it>
% Last updated: 2023-08-10

The solution of a linear system $Ax = b$, where $A$ is a square matrix
of size $n \times n$ in upper triangular form and $b$ is a vector of
size $n$, can be computed through _Gaussian elimination_ using the
following code fragment:

```C
    for (int i=n-1; i>=0; i--) {
        x[i] = b[i];
        for (int j=i+1; j<n; j++) {
            x[i] -= A[i*n + j]*x[j];
        }
        x[i] /= A[i*n + i];
    }

```

The idea is to start from the last equation to find $x_{n-1}$, and the
proceed backwards by substituting the known values $x_{i+1}, x_{i+2},
\ldots, x_{n-1}$ to compute $x_i$ as:

$$
x_i = b_i - \sum_{j = i+1}^{n-1} A_{ij} x_j \qquad i = n-1, n-2, \ldots, 0
$$

For example, consider the following syetsm of equations in upper
triangular form:

$$
\renewcommand\arraystretch{1.25}
\left\{
\begin{array}{rcrcrcl}
   2x_0 & + &  3x_1 & - & 2x_2 & = & 10 \\
        &   & -4x_1 & + &  x_2 & = &  6 \\
        &   &       &   & 5x_2 & = & 13
\end{array}
\right.
$$

From the last equation one immediately gets $x_2 = 13/5$. We
substitute this value into the second equation, to get

$$
x_1 = (6 - 13/5) / (-4)
$$

and finally, knowing both $x_1$ and $x_2$, we can compute $x_0$ by
substituting the known values into the first equation.

The goal of this exercise is to parallelize the code fragment below
using suitable MPI constructs.

To compile:

        mpicc -std=c99 -Wall -Wpedantic mpi-gaussian-elimination.c -o mpi-gaussian-elimination

To execute:

        mpirun -n P ./mpi-gaussian-elimination N

Example:

        mpirun -n 4 ./mpi-gaussian-elimination 10

## Files

- [mpi-gaussian-elimination.c](mpi-gaussian-elimination.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "hpc.h"

/**
 * Solve the linear system Ax = b, where A is a square matrix of size
 * n x n in upper triangular form.
 */
#ifdef SERIAL
void solve (const float *A, const float *b, float *x, int n ){

    for (int i=n-1; i>=0; i--){
        x[i] = b[i];
        for (int j=i+1; j<n; j++) {
            x[i] -= A[i*n + j]*x[j];
        }
        x[i] /= A[i*n + i];
    }

}
#endif

void init (float *A, float *b, int n ) {
    const float EPSILON = 1e-5;
    for (int i = 0; i < n; i++) {
        b[i] = i;
        for (int j = 0; j < n; j++) {
            if (i > j) {
                A[i*n + j] = 0;
            } else {
                A[i*n + j] = i + j + 1;
            }
        }
        /* ensures that matrix A is non-singular */
        assert( fabs(A[i*n + i]) > EPSILON );
    }
}

/**
 * Returns nonzero iff Ax = b within some tolerante EPSILON
 */
int check_ok( const float *A, const float *b, const float *x, int n ) {
    const float EPSILON = 1e-3;
    for (int i = 0; i < n; i++) {
        assert( ! isnan(x[i]) );
        assert( ! isinf(x[i]) );
        float lhs = 0; // left-hand side value of equation i
        for (int j = 0; j < n; j++) {
            lhs += x[j] * A[i*n + j];
        }
        if (fabs(lhs - b[i]) > EPSILON) {
            fprintf(stderr, "ERROR equation %d: LHS = %f, b[%d] = %f\n", i, lhs, i, b[i]);
            return 0;
        }
    }
    return 1;
}

void print( const float *A, const float *b, const float *x, int n ) {
    printf("A[][] =\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", A[i*n+j]);
        }
        printf("\n");
    }
    printf("\n\nb[] = ");
    for (int i = 0; i < n; i++) {
        printf("%f ", b[i]);
    }
    printf("\n\nx[] = ");
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n\n");
}

int main( int argc, char *argv[] ) {
    int my_rank, comm_sz, n = 10;
    float *A, *b, *x;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    const double tstart = hpc_gettime();
#ifdef SERIAL

    if (my_rank == 0){

        A = (float*) malloc(n * n * sizeof(*A)); 
        assert(A != NULL);
        b = (float*) malloc(n * sizeof(*b)); 
        assert(b != NULL);
        x = (float*) malloc(n * sizeof(*x)); 
        assert(x != NULL);

        printf("Initializing...\n");
        init(A, b, n);
        printf("Solving...\n");
        solve(A, b, x, n);
    }
#else
     MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        A = (float*) malloc(n * n * sizeof(*A)); 
        assert(A != NULL);
    }

    b = (float*) malloc(n * sizeof(*b)); 
    assert(b != NULL);
    x = (float*) malloc(n * sizeof(*x)); 
    assert(x != NULL);

    if (my_rank == 0) {
        printf("Initializing...\n");
        init(A, b, n);
        printf("Solving...\n");
    }
    MPI_Bcast(b, n, MPI_FLOAT, 0, MPI_COMM_WORLD); 

    int *sendcounts = (int*) malloc(comm_sz * sizeof(*sendcounts));
    int *displs = (int*) malloc(comm_sz * sizeof(*displs));
    for (int i = 0; i < comm_sz; i++) {
        const int istart = n * i /comm_sz;
        const int iend = n * (i+1) /comm_sz;
        sendcounts[i] = iend - istart;
        displs[i] = istart;
    }

    float *local_A = (float*) malloc(sendcounts[my_rank] * n * sizeof(float));
    assert(local_A != NULL);

    float xi;
    int start = n * my_rank /comm_sz;
    int end = n * (my_rank+1) /comm_sz;

    for (int i = n - 1; i >= 0; i--) {
        xi = 0;
        /* Scatters row A[i] in parts to all processes */
        MPI_Scatterv(&A[i * n], sendcounts, displs, MPI_FLOAT, &local_A[0], sendcounts[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
        for (int j = i + 1; j < n; j++) {
            if (j >= start && j < end) {  
                /* Each process calculates local x[i] */      
                xi -= local_A[j - start] * x[j]; 
            }
        }
        /* The result is stored in process 0 that manages the last two operations */
        MPI_Reduce(&xi, &x[i], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(my_rank == 0){
            x[i] += b[i];
            x[i] /= A[i * n + i];
        }
        /* Distributes the final result to each process */    
        MPI_Bcast(&x[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
    }

#endif
    const double elapsed = hpc_gettime() - tstart;
    if(my_rank == 0){
      /*  print(A, b, x, n);*/

        if (check_ok(A, b, x, n)) {
            printf("Check OK\n");
            fprintf(stderr,"\nExecution time %f seconds\n", elapsed);
        } else {
            printf("Check FAILED\n");
        }

        free(A);
        free(b);
    }

#ifdef SERIAL
    free(x);
#else
    free(x);
    free(local_A);
    free(sendcounts);
    free(displs);
#endif
    MPI_Finalize();
    return EXIT_SUCCESS;
}
