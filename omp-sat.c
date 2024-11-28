/****************************************************************************
 *
 * omp-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023, 2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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
% HPC - Brute-force SAT solver
% [Moreno Marzolla](https://www.moreno.marzolla.name/)
% Last modified: 2024-11-23

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
31 Boolean variables, so that an assignment can be encoded as an `int`
value.

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

        gcc -fopenmp -Wall -Wpedantic omp-sat.c -o omp-sat

To execute:

        ./omp-sat < queens-05.cnf

## Files

- [omp-sat.c](omp-sat.c)
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>
***/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <omp.h>

/* MAXLITERALS must be at most (bit width of int) - 2 */
#define MAXLITERALS 30
/* MAXCLAUSES must be a power of two */
#define MAXCLAUSES 512

typedef struct {
    int x[MAXCLAUSES], nx[MAXCLAUSES];
    int nlit;
    int nclauses;
} problem_t;

int max(int a, int b)
{
    return (a>b ? a : b);
}

int abs(int x)
{
    return (x>=0 ? x : -x);
}

void print_solution(const problem_t *p, int v)
{
#if 0
#ifndef SERIAL
#pragma omp critical
    {
#endif
    const int NLIT = p->nlit;
    int mask = 1;
    for (int i=1; i<=NLIT; i++) {
        printf("x_%d=%d ", i, (v & mask) != 0);
        mask = mask << 1;
    }
    printf("\n");
#ifndef SERIAL
    }
#endif
#endif
}

/**
 * Evaluate problem `p` in conjunctive normal form by setting the i-th
 * variable to the value of bit i of `v` (bit 0 is the rightmost
 * bit). Returns the value of the boolean expression encoded by `p`.
 */
bool eval(const problem_t *p, int v)
{
    bool result = true;
    for (int c=0; c < p->nclauses && result; c++) {
        const bool term = (v & p->x[c]) | (~v & p->nx[c]);
        result &= term;
    }
    if (result) print_solution(p, v);
    return result;
}

/**
 * Returns the number of solutions to the SAT problem `p`.
 */
int sat( const problem_t *p)
{
    const int NLIT = p->nlit;
    const unsigned int MAX_VALUE = (1u << NLIT) - 1;
    int nsat = 0;

    /* `cur_value` should be of type `unsigned int` since at the end
       of the loop it might overflow the `int` type if NLIT==31 */
#ifndef SERIAL
#pragma omp parallel for default(none) shared(p, MAX_VALUE) reduction(+:nsat)
#endif
    for (unsigned int cur_value=0; cur_value<=MAX_VALUE; cur_value++) {
        nsat += eval(p, cur_value);
    }
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

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input\n", argv[0]);
        return EXIT_FAILURE;
    }
    load_dimacs(stdin, &p);
    pretty_print(&p);
    const double tstart = omp_get_wtime();
    int nsolutions = sat(&p);
    const double elapsed = omp_get_wtime() - tstart;
    printf("%d solutions in %f seconds\n", nsolutions, elapsed);
    return EXIT_SUCCESS;
}
