/****************************************************************************
 *
 * omp-sat.c - Brute-force SAT solver
 *
 * Copyright (C) 2018, 2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
% Last updated: 2023-03-27

To compile:

        gcc -fopenmp -Wall -Wpedantic omp-sat.c -o omp-sat

To execute:

        ./omp-sat < sat.cnf

## Files

- [omp-sat.c](omp-sat.c)
- Some input files: <queens-05.cnf>, <uf20-01.cnf>, <uf20-077.cnf>
***/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define MAXLITERALS 30
#define MAXCLAUSES 300

typedef struct {
    int lit[MAXCLAUSES][MAXLITERALS];
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

/**
 * Evaluate problem |p| in conjunctive normal form by setting the i-th
 * variable to v[i]. Returns the value of the boolean expression
 * encoded by |p|.
 */
bool eval(const problem_t* p, const bool *v)
{
    int c, l;
    for (c=0; c < p->nclauses; c++) {
        bool term = false;
        for (l=0; p->lit[c][l]; l++) {
            const int x = p->lit[c][l];
            if (x > 0) {
                term |= v[x];
            } else {
                term |= !v[-x];
            }
        }
        if ( false == term ) { return false; }
    }
    return true;
}

/**
 * Pretty-prints problem |p|
 */
void pretty_print( const problem_t *p )
{
    int c, l;
    for (c=0; (c < MAXCLAUSES) && p->lit[c][0]; c++) {
        printf("( ");
        for (l=0; (l < MAXLITERALS) && p->lit[c][l]; l++) {
            if (p->lit[c][l] > 0 ) {
                printf("x_%d ", p->lit[c][l]);
            } else {
                printf("¬x_%d ", -(p->lit[c][l]));
            }
            if ((l < MAXLITERALS-1) && p->lit[c][l+1] ) {
                printf("∨ ");
            }
        }
        printf(")");
        if ((c < MAXCLAUSES-1) && p->lit[c+1][0]) {
            printf(" ∧");
        }
        printf("\n");
    }
}

/**
 * Load a DIMACS CNF file |f| and initialize problem |p|.  The DIMACS
 * CNF format specification fan be found at
 * https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps
 */
void load_dimacs( FILE *f, problem_t *p )
{
    int result;
    int c, l, val;
    int prob_c, prob_l;

    /* Set all literals to false */
    for (c=0; c<MAXCLAUSES; c++) {
        for (l=0; l<MAXLITERALS; l++) {
            p->lit[c][l] = 0;
        }
    }
    p->nlit = -1;
    c = l = 0;
    /* From
       https://github.com/marijnheule/march-SAT-solver/blob/master/parser.c */
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
            /* New clause */
            l = 0;
            c++;
        } else {
            /* New literal */
            p->lit[c][l] = val;
            p->nlit = max(p->nlit, abs(val));
            l++;
        }
    }
    p->nclauses = c;
    fprintf(stderr, "DIMACS CNF files: %d clauses, %d literals\n", c, p->nlit);
}

int sat( const problem_t *p)
{
    const int nlit = p->nlit;
    const uint32_t maxval = (1u << nlit) - 1;
    uint32_t cur_value;
    int nsat = 0;
    bool v[MAXLITERALS];

    assert( sizeof(cur_value) < nlit );
#if __GNUC__ < 9
#pragma omp parallel for default(none) private(v) shared(p) reduction(+:nsat)
#else
#pragma omp parallel for default(none) private(v) shared(p, maxval) reduction(+:nsat)
#endif
    for (cur_value=0; cur_value<maxval; cur_value++) {
        /* convert cur in binary */
        int idx = 1;
        for (uint32_t mask=1; mask < maxval; mask = mask << 1) {
            v[idx] = ((cur_value & mask) != 0);
            idx++;
        }
        nsat += eval(p, v);
    }
    return nsat;
}

int main( void )
{
    problem_t p;

    load_dimacs(stdin, &p);
    pretty_print(&p);
    const double tstart = omp_get_wtime();
    int nsolutions = sat(&p);
    const double elapsed = omp_get_wtime() - tstart;
    printf("%d solutions in %f seconds\n", nsolutions, elapsed);
    return EXIT_SUCCESS;
}