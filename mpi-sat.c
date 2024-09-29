/****************************************************************************
 *
 * mpi-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023, 2024 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% HPC - Brute-force SAT solver
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last modified: 2024-09-29

A SAT problem is represented as a pair of integer arrays `x[]` and
`nx[]` of length `nclauses`. `x[i]` and `nx[i]` represents the _i_-th
clause as follows. The binary digits of `x[i]` are the coefficients of
the terms that are _true_ in the _i_-th clause; the binary digits of
`nx[i]` are the literals that are _false_ in the _i_-th clause.  For
example, if the _i_-th clause is:

$$
x_1 \vee x_3 \vee \neg x_4 \vee \neg x_5 \vee x_7
$$

then it is represented as:

         x[i] = 1000101_2 = 69_10
        nx[i] = 0011000_2 = 24_10

(here _2 and _10 denote base two and ten, respectively). Note that
literals are indexed starting from one; the coefficient of $x_1$ is
encoded in the rightmost bit. The representation above allows at most
`8*sizeof(int)-1` literals using the `int` datatype, which increases
to `8*sizeof(uint32_t)` using `uint32_t`.

The representation has the advantage that evaluating a term can be
done in constant time. Let the binary digits of `v` represent the
assignment of values to literals.  Then, the term is true if and only
if

        (v & x[i]) | (~v & nx[i])

is nonzero.

For example, let us consider the assignment

$$
x_1 = x_3 = x_4 = x_5 = 0 \\
x_2 = x_6 = x_7 = 1
$$

This assignment can be encoded as an integer `v` whose binary
representation is:

        v = (x_7 x_6 x_5 x_4 x_3 x_2 x_1)_2 = 1100010_2

and using the values of `x` and `nx` above, we have:

```
(v & x[i]) | (~v & nx[i]) =
(1100010 & 1000101) | (0011101 & 0011000) ==
1000000 | 0011000 ==
1011000
```

which is nonzero, and hence with the assignment above the term is
true.

To compile:

        mpicc -Wall -Wpedantic mpi-sat.c -o mpi-sat

To execute:

        mpirun -n 4 ./mpi-sat < queens-05.cnf

## Files

- [mpi-sat.c](mpi-sat.c)
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>
***/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

/* MAXLITERALS must be at most (bit width of int) - 2 */
#define MAXLITERALS 30
/* MAXCLAUSES must be a power of two */
#define MAXCLAUSES 512

typedef struct {
    int x[MAXCLAUSES], nx[MAXCLAUSES];
    int nlit;
    int nclauses;
} problem_t;

int my_rank, comm_sz;

int max(int a, int b)
{
    return (a>b ? a : b);
}

int abs(int x)
{
    return (x>=0 ? x : -x);
}

/**
 * Evaluate problem `p` in conjunctive normal form by setting the i-th
 * variable to the value of bit (i+1) of `v` (bit 0 is the leftmost
 * bit, which is not used). Returns the value of the boolean
 * expression encoded by `p`.
 */
bool eval(const problem_t* p, const int v)
{
    bool result = true;
    for (int c=0; c < p->nclauses && result; c++) {
        const bool term = (v & p->x[c]) | (~v & p->nx[c]);
        result &= term;
    }
    return result;
}

/**
 * Returns the number of solutions to the SAT problem `p`.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const int MAX_VALUE = (1 << NLIT) - 1;
    const int my_start = ((MAX_VALUE+1)*my_rank) / comm_sz;
    const int my_end = ((MAX_VALUE+1)*(my_rank+1)) / comm_sz;

    int my_nsat = 0, nsat;

    for (int cur_value=my_start; cur_value < my_end; cur_value++) {
        my_nsat += eval(p, cur_value);
    }

    MPI_Reduce(&my_nsat,        // sendbuf
               &nsat,           // recvbuf
               1,               // count
               MPI_INT,         // datatype
               MPI_SUM,         // op
               0,               // root
               MPI_COMM_WORLD);
    return nsat;
}

/**
 * Pretty-print problem `p`
 */
void pretty_print( const problem_t *p )
{
    for (int c=0; c < p->nclauses; c++) {
        printf("( ");
        int x = p->x[c];
        int nx = p->nx[c];
        for (int l=0, printed=0; l < MAXLITERALS; l++) {
            if (x & 1) {
                printf("%sx_%d", printed ? " ∨ " : "", l);
                printed = 1;
            } else if (nx & 1) {
                printf("%s¬x_%d", printed ? " ∨ " : "", l);
                printed = 1;
            }
            x = x >> 1;
            nx = nx >> 1;
        }
        printf(" )");
        if (c < p->nclauses - 1) {
            printf(" ∧");
        }
        printf("\n");
    }
}

/**
 * Load a DIMACS CNF file `f` and initialize problem `p`.  The DIMACS
 * CNF format specification can be found at
 * <https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps>
 */
void load_dimacs( FILE *f, problem_t *p )
{
    int result;
    int c, l, val;
    int prob_c, prob_l;

    /* Clear all bitmasks */
    for (c=0; c<MAXCLAUSES; c++) {
        p->x[c] = p->nx[c] = 0;
    }
    p->nlit = -1;
    c = l = 0;
    /* From
       <https://github.com/marijnheule/march-SAT-solver/blob/master/parser.c> */
    do {
        result = fscanf(f, " p cnf %i %i \n", &prob_l, &prob_c);
        if ( result > 0 && result != EOF )
            break;
        result = fscanf(f, "%*s\n");
    } while( result != 2 && result != EOF );

    if ( prob_l > MAXLITERALS-1 ) {
        fprintf(stderr, "FATAL: too many literals (%d); please set MAXLITERALS to at least %d\n", prob_l, prob_l+1);
        exit(EXIT_FAILURE);
    }
    if ( prob_c > MAXCLAUSES-1 ) {
        fprintf(stderr, "FATAL: too many clauses (%d); please set MAXCLAUSES to at least %d\n", prob_c, prob_c+1);
        exit(EXIT_FAILURE);
    }
    while (fscanf(f, "%d", &val) == 1) {
        if (val == 0) {
            /* Check the previous clause for consistency */
            assert( (p->x[c] & p->nx[c]) == 0 );
            /* New clause */
            l = 0;
            c++;
        } else {
            /* New literal */
            if (val > 0) {
                p->x[c] |= (1 << (val-1));
            } else {
                p->nx[c] |= (1 << -(val+1));
            }
            p->nlit = max(p->nlit, abs(val));
            l++;
        }
    }
    p->nclauses = c;
    fprintf(stderr, "DIMACS CNF files: %d clauses, %d literals\n", c, p->nlit);
}

int main( int argc, char *argv[] )
{
    problem_t p;

    assert(MAXLITERALS <= 8*sizeof(int)-1);
    assert((MAXCLAUSES & (MAXCLAUSES-1)) == 0); /* "bit hack" to check whether MAXCLAUSES is a power of two */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc != 1 && my_rank == 0) {
        fprintf(stderr, "Usage: %s < input\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (my_rank == 0) {
        load_dimacs(stdin, &p);
        pretty_print(&p);
    }

    MPI_Bcast(&p.x,             // buffer
              MAXCLAUSES,       // count
              MPI_INT,          // datatype
              0,                // root
              MPI_COMM_WORLD);

    MPI_Bcast(&p.nx,            // buffer
              MAXCLAUSES,       // count
              MPI_INT,          // datatype
              0,                // root
              MPI_COMM_WORLD);

    MPI_Bcast(&p.nlit,          // buffer
              1,                // count
              MPI_INT,          // datatype
              0,                // root
              MPI_COMM_WORLD);

    MPI_Bcast(&p.nclauses,      // buffer
              1,                // count
              MPI_INT,          // datatype
              0,                // root
              MPI_COMM_WORLD);

    const double tstart = MPI_Wtime();
    int nsolutions = sat(&p);
    const double elapsed = MPI_Wtime() - tstart;

    if (my_rank == 0)
        printf("%d solutions in %f seconds\n", nsolutions, elapsed);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
