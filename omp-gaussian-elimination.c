/****************************************************************************
 *
 * omp-gaussian-elimination.c - Solve systems of linear equations in upper triangular form
 *
 * Copyright (C) 2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated: 2023-01-13

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

The goal fo this exercise is to parallelize the code fragment below
using suitable OpenMP constructs.

# Files

- [omp-gaussian-elimination.c](omp-gaussian-elimination.c)

***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/**
 * Solve the linear system Ax = b, where A is a square matrix of size
 * n x n in upper triangular form.
 */
void solve( const float *A, const float *b, float *x, int n )
{
#ifdef SERIAL
    for (int i=n-1; i>=0; i--) {
        x[i] = b[i];
        for (int j=i+1; j<n; j++) {
            x[i] -= A[i*n + j]*x[j];
        }
        x[i] /= A[i*n + i];
        /* HINT: you might want to rewrite the outer loop body as
           follows:

           float xi = b[i];
           for (int j=i+1; j<n; j++) {
             xi -= A[i*n + j]*x[j];
           }
           x[i] = xi / A[i*n + i];

           Although this version ios more verbose and definitely less
           clear, it shows that a known parallel pattern applies to
           the inner loop...
        */
    }
#else
    for (int i=n-1; i>=0; i--) {
        float xi = b[i];
#pragma omp parallel for default(none) shared(A,i,n,x) reduction(-:xi)
        for (int j=i+1; j<n; j++) {
            xi -= A[i*n + j]*x[j];
        }
        x[i] = xi / A[i*n + i];
    }
#endif
}

void init( float *A, float *b, int n )
{
    const float EPSILON = 1e-5;
    for (int i=0; i<n; i++) {
        b[i] = i;
        for (int j=0; j<n; j++) {
            if (i > j) {
                A[i*n + j] = 0;
            } else {
                A[i*n + j] = i+j + 1;
            }
        }
        /* ensures that matrix A is non-singular */
        assert( fabs(A[i*n + i]) > EPSILON );
    }
}

/**
 * Returns nonzero iff Ax = b within some tolerante EPSILON
 */
int check_ok( const float *A, const float *b, const float *x, int n )
{
    const float EPSILON = 1e-3;
    for (int i=0; i<n; i++) {
        assert( ! isnan(x[i]) );
        assert( ! isinf(x[i]) );
        float lhs = 0; // left-hand side value of equation i
        for (int j=0; j<n; j++) {
            lhs += x[j] * A[i*n + j];
        }
        if (fabs(lhs - b[i]) > EPSILON) {
            fprintf(stderr, "ERROR equation %d: LHS = %f, b[%d] = %f\n", i, lhs, i, b[i]);
            return 0;
        }
    }
    return 1;
}

void print( const float *A, const float *b, const float *x, int n )
{
    printf("A[][] =\n");
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            printf("%.2f ", A[i*n+j]);
        }
        printf("\n");
    }
    printf("\n\nb[] = ");
    for (int i=0; i<n; i++) {
        printf("%f ", b[i]);
    }
    printf("\n\nx[] = ");
    for (int i=0; i<n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n\n");
}

int main( int argc, const char *argv[] )
{
    int n = 10;
    float *A, *b, *x;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    A = (float*)malloc(n * n * sizeof(*A)); assert(A != NULL);
    b = (float*)malloc(n * sizeof(*b)); assert(b != NULL);
    x = (float*)malloc(n * sizeof(*x)); assert(x != NULL);

    printf("Initializing...\n");
    init(A, b, n);
    printf("Solving...\n");
    solve(A, b, x, n);
    print(A, b, x, n);
    if (check_ok(A, b, x, n)) {
        printf("Check OK\n");
    } else {
        printf("Check FAILED\n");
    }
    free(A);
    free(b);
    free(x);
    return EXIT_SUCCESS;
}