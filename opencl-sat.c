/****************************************************************************
 *
 * opencl-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023, 2024 Moreno Marzolla
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
% Brute-force SAT solver
% [Moreno Marzolla](https://www.unibo.it/sitoweb/moreno.marzolla)
% Last updated: 2024-11-29

The Boolean Satisfability Problem (SAT Problem) asks whether there
exists an assignment of boolean variables $x_1, \ldots, x_n$ that
makes a given boolean expression true. In this exercise we consider
Boolean expressions in _Conjunctive Normal Form_ (CNF), which means
that the expression is the conjunction (logical _and_) of clauses
which in turn are the disjunction (logical _or_) of (possibly negated)
variables. For example, the following expression with variables $x_1,
x_2, x_3, x_4$ is in CNF:

$$
(x_1 \vee \neg x_2 \vee x_4) \wedge
(\neg x_2 \vee x_3) \wedge
(\neg x_1 \vee x_2 \vee \neg x_4)
$$

The expression above has three _clauses_, namely, $(x_1 \vee \neg x_2
\vee x_4)$, $(\neg x_2 \vee x_3)$ and $(\neg x_1 \vee x_2 \vee \neg
x_4)$.

An assignment of $n$ Boolean variables $x_1, x_2, \ldots, x_n$ can be
encoded as a binary value $v = (x_n \cdots x_2 x_1)$. In this exercise
we develop a parallel SAT solver that tries all $2^n$ assignments, and
counts the number of solutions, i.e., the number of assignments that
make the Boolean expression true.

To make the approach less inefficient, it is important to use a
suitable representation of the Boolean expression, so that it can be
evaluated quickly. To this aim, let us assume that there are at most
30 Boolean variables, so that an assignment can be encoded as an `int`
value without overflow (see the code).

We represent a SAT problem as a pair of integer arrays `x[]` and
`nx[]` of length `nclauses`. `x[i]` and `nx[i]` represents the _i_-th
clause as follows. The binary digits of `x[i]` correspond to the
variables that are not negated in the _i_-th clause; the binary digits
of `nx[i]` correspond to the variables that are negated in the _i_-th
clause.

For example, if the _i_-th clause is:

$$
x_1 \vee x_3 \vee \neg x_4 \vee \neg x_5 \vee x_7
$$

then $x_1, x_3, x_7$ are not negated, and therefore the first, third
and seventh bit of `x[i]` will be set to one (the first bit is the
rightmost one); similarly, $x_4, x_5$ are negated, and therefore the
fourth and fifth bits of `nx[i]` will be set to one:

         x[i] = 1000101_2 = 69_10
        nx[i] = 0011000_2 = 24_10

(here _2 and _10 denote base two and ten, respectively). Note that
literals are indexed starting from one. This representation has the
advantage that evaluating a clause can be done in constant time. Let
the binary digits of an intecer `v` represent the assignment of values
to literals.  Then, clause _i_ is true if and only if

        (v & x[i]) | (~v & nx[i])

is nonzero.

For example, let us consider the assignment

$$
x_1 = x_3 = x_4 = x_5 = 0 \qquad
x_2 = x_6 = x_7 = 1
$$

This assignment can be encoded as an integer `v` whose binary
representation is:

$$
        v = (x_7 x_6 x_5 x_4 x_3 x_2 x_1)_2 = 1100010_2
$$
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

        gcc -Wall -Wpedantic opencl-sat.c simpleCL.c -o opencl-sat -LOpenCL

To execute:

        ./opencl-sat < sat.cnf

## Files

- [opencl-sat.c](opencl-sat.c)
- [simpleCL.c](simpleCL.c) [simpleCL.h](simpleCL.h) [hpc.h](hpc.h).
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>7

***/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "simpleCL.h"
#include "hpc.h"

#define MAXLITERALS (8*sizeof(int) - 2)
#define MAXCLAUSES 512

typedef struct {
    int x[MAXCLAUSES], nx[MAXCLAUSES];
    int nlit;
    int nclauses;
} problem_t;

#ifndef SERIAL
sclKernel eval_kernel;
#endif

int max(int a, int b)
{
    return (a>b ? a : b);
}

int abs(int x)
{
    return (x>=0 ? x : -x);
}

#ifdef SERIAL
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
    int nsat = 0;

    for (int cur_value=0; cur_value<=MAX_VALUE; cur_value++) {
        nsat += eval(p, cur_value);
    }
    return nsat;
}
#else
/**
 * OpenCL implementation of a brute-force SAT solver. It uses 1D
 * workgroup of 1D work-items; each workgroup has `p->nclauses`
 * work-items and evaluates a clause. Different work-items evaluate
 * different clauses in parallel. We can not launch `MAX_VALUE`
 * work-items (one for each possible combination of assignments),
 * since that might exceed hardware limits. Therefore, multiple kernel
 * launches are required in the "for" loop below.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const int NCLAUSES = p->nclauses;
    const int MAX_VALUE = (1 << NLIT) - 1;
    const sclDim BLOCK = DIM1(SCL_DEFAULT_WG_SIZE);
    const int CHUNK_SIZE = SCL_DEFAULT_WG_SIZE * 2048; /* you might need to change this depending on your hardware */
    const sclDim GRID = DIM1(CHUNK_SIZE);

    int nsat = 0;
    cl_mem d_nsat, d_x, d_nx;

    d_x = sclMallocCopy(NCLAUSES * sizeof(*(p->x)), (void*)(p->x), CL_MEM_READ_ONLY);
    d_nx = sclMallocCopy(NCLAUSES * sizeof(*(p->nx)), (void*)(p->nx), CL_MEM_READ_ONLY);
    d_nsat = sclMallocCopy(sizeof(nsat), &nsat, CL_MEM_READ_WRITE);

    for (int cur_value=0; cur_value<=MAX_VALUE; cur_value += CHUNK_SIZE) {
        sclSetArgsEnqueueKernel(eval_kernel,
                                GRID, BLOCK,
                                ":b :b :d :d :d :b",
                                d_x,
                                d_nx,
                                p->nlit,
                                p->nclauses,
                                cur_value,
                                d_nsat);
    }
    sclMemcpyDeviceToHost(&nsat, d_nsat, sizeof(nsat));

    sclFree(d_nsat);
    sclFree(d_x);
    sclFree(d_nx);
    return nsat;
}
#endif

/**
 * Pretty-prints problem `p`
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
    int ret, nc, nl;

    /* Clear all bitmasks */
    for (int c=0; c<MAXCLAUSES; c++) {
        p->x[c] = p->nx[c] = 0;
    }

    do {
        ret = fscanf(f, " p cnf %i %i \n", &nl, &nc);
        if ( ret > 0 && ret != EOF )
            break;
        ret = fscanf(f, "%*s\n");
    } while( ret != 2 && ret != EOF );

    if ( nl > MAXLITERALS-1 ) {
        fprintf(stderr, "FATAL: too many literals (%d); please set MAXLITERALS to at least %d\n", nl, nl+1);
        exit(EXIT_FAILURE);
    }
    if ( nc > MAXCLAUSES-1 ) {
        fprintf(stderr, "FATAL: too many clauses (%d); please set MAXCLAUSES to at least %d\n", nc, nc+1);
        exit(EXIT_FAILURE);
    }
    p->nlit = -1; /* we compute the number of literals from those actually used in the file */
    p->nclauses = nc;
    for (int c=0; c<nc; c++) {
        int val;
        do {
            ret = fscanf(f, "%d", &val);
            assert(ret == 1);
            if (val)  {
                if (val > 0) {
                    p->x[c] |= (1 << (val-1));
                } else {
                    p->nx[c] |= (1 << -(val+1));
                }
                p->nlit = max(p->nlit, abs(val));
            }
        } while (val);
        assert( (p->x[c] & p->nx[c]) == 0 );
    }
    fprintf(stderr, "DIMACS CNF files: %d clauses, %d literals\n", nc, p->nlit);
}

int main( int argc, char *argv[] )
{
    problem_t p;

    assert(MAXLITERALS <= 8*sizeof(int)-2);

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input\n", argv[0]);
        return EXIT_FAILURE;
    }
#ifndef SERIAL

    sclInitFromFile("opencl-sat.cl");
    eval_kernel = sclCreateKernel("eval_kernel");
#endif

    load_dimacs(stdin, &p);
    pretty_print(&p);
    const double tstart = hpc_gettime();
    int nsolutions = sat(&p);
    const double elapsed = hpc_gettime() - tstart;
    printf("%d solutions in %f seconds\n", nsolutions, elapsed);
#ifndef SERIAL
    sclFinalize();
#endif
    return EXIT_SUCCESS;
}
